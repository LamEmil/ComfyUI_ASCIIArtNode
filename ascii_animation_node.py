import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class ASCIIAnimationGeneratorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        default_charset = " .'`^\":;Il!i~_?[{1(|/fjxnvzXYJCLQ0Zmwqdkh*#M&8%B@$"
        
        return {
            "required": {
                "image": ("IMAGE",),
                "char_width": ("INT", {"default": 80, "min": 10, "max": 1000, "step": 10, "display": "slider"}), # Max width reduced for performance
                "font_path": ("STRING", {"default": "cour.ttf", "multiline": False}), 
                "font_size": ("INT", {"default": 15, "min": 5, "max": 100, "step": 1, "display": "slider"}),
                "ascii_charset": ("STRING", {"default": default_charset, "multiline": True}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "text_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}),
                "chars_per_frame": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display":"slider"}), # New: control typing speed
            }
        }

    RETURN_TYPES = ("IMAGE",) # Output is a batch of images
    RETURN_NAMES = ("animated_ascii_frames",)
    FUNCTION = "generate_ascii_animation"
    CATEGORY = "image/animation" # Or image/art

    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        """Converts a HWC PyTorch tensor (0-1 float) from the input batch to a PIL Image."""
        img_np = tensor_image.cpu().numpy()
        # Input tensor_image is expected to be HWC (from image[0])
        img_np = (img_np * 255).astype(np.uint8)
        
        if img_np.shape[-1] == 1: # Grayscale
            return Image.fromarray(img_np.squeeze(-1), 'L')
        else: # RGB or RGBA
            return Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA')

    def pil_to_tensor_frame(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a HWC PyTorch tensor (0-1 float) for a single animation frame."""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if img_np.ndim == 2: # Grayscale image, add channel dimension
            img_np = np.expand_dims(img_np, axis=2)
        
        tensor_image = torch.from_numpy(img_np) # HWC
        return tensor_image

    def generate_ascii_animation(self, image: torch.Tensor, char_width: int, font_path: str, 
                                 font_size: int, ascii_charset: str, background_color: str, 
                                 text_color: str, invert_brightness_mapping: bool,
                                 chars_per_frame: int):
        
        # 0. Basic Input Validation
        if not ascii_charset:
            raise ValueError("ASCII charset cannot be empty.")
        if chars_per_frame < 1:
            chars_per_frame = 1

        # 1. Convert Input Tensor to PIL Image (use the first image in the batch)
        if image.ndim == 3: # Should be (B,H,W,C), if (H,W,C) add batch
            image = image.unsqueeze(0)
        
        input_pil_image = self.tensor_to_pil(image[0]) # image[0] is HWC
        original_width, original_height = input_pil_image.size

        # 2. Convert PIL Image to Full ASCII Representation (list of strings)
        aspect_ratio = original_height / original_width
        # Character cell aspect ratio adjustment (0.5 means chars are ~twice as tall as wide)
        char_height = max(1, int(char_width * aspect_ratio * 0.5)) 

        resized_image_for_ascii = input_pil_image.resize((char_width, char_height), Image.Resampling.LANCZOS)
        grayscale_image = resized_image_for_ascii.convert("L")

        full_ascii_lines = []
        pixels = grayscale_image.load()
        for y_idx in range(char_height):
            line_of_text = ""
            for x_idx in range(char_width):
                brightness = pixels[x_idx, y_idx]
                if invert_brightness_mapping:
                    brightness = 255 - brightness
                char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                line_of_text += ascii_charset[char_index]
            full_ascii_lines.append(line_of_text)

        if not any(full_ascii_lines): # If all lines are empty
            full_ascii_lines = [" "] # Ensure at least one space to avoid errors later

        # 3. Prepare for Rendering (Font, Colors, Base Text Dimensions)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try: font = ImageFont.load_default(font_size=font_size)
            except AttributeError: font = ImageFont.load_default()
            except TypeError: font = ImageFont.load_default()
        
        try:
            bg_color_rgb = ImageColor.getrgb(background_color)
            text_color_rgb = ImageColor.getrgb(text_color)
        except ValueError:
            print(f"Warning: Invalid color string. Using black background/white text.")
            bg_color_rgb = (0,0,0)
            text_color_rgb = (255,255,255)

        # Calculate dimensions of the text block if all ASCII art was rendered
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
        max_text_pixel_width = 0
        for line in full_ascii_lines:
            try: line_bbox = temp_draw.textbbox((0,0), line, font=font); width = line_bbox[2] - line_bbox[0]
            except AttributeError: width, _ = temp_draw.textsize(line, font=font) # Older Pillow
            max_text_pixel_width = max(max_text_pixel_width, width)
        
        try: metrics_bbox = temp_draw.textbbox((0,0), "My", font=font); line_pixel_height = metrics_bbox[3] - metrics_bbox[1]
        except AttributeError: _, line_pixel_height = temp_draw.textsize("My", font=font) # Older Pillow

        if line_pixel_height == 0 and font_size > 0: line_pixel_height = int(font_size * 1.2)
        if max_text_pixel_width == 0: max_text_pixel_width = char_width * font_size // 2 
        if line_pixel_height == 0: line_pixel_height = font_size

        text_render_width = max(1, max_text_pixel_width)
        text_render_height = max(1, len(full_ascii_lines) * line_pixel_height)

        # 4. Generate Animation Frames
        output_frame_tensors = []
        total_chars_to_type = sum(len(line) for line in full_ascii_lines)
        chars_typed_so_far = 0
        
        # Always generate at least one frame, even if it's just the background
        # or the first few characters if total_chars_to_type is small.
        
        while chars_typed_so_far <= total_chars_to_type:
            current_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_frame = ImageDraw.Draw(current_frame_pil)
            
            chars_drawn_this_frame_total = 0
            temp_chars_typed_count = 0 # Relative to start of full_ascii_lines

            for line_idx, line_content in enumerate(full_ascii_lines):
                chars_to_draw_on_this_line = 0
                if temp_chars_typed_count < chars_typed_so_far:
                    remaining_to_type_on_line = chars_typed_so_far - temp_chars_typed_count
                    chars_to_draw_on_this_line = min(len(line_content), remaining_to_type_on_line)
                
                if chars_to_draw_on_this_line > 0:
                    draw_frame.text((0, line_idx * line_pixel_height), 
                                    line_content[:chars_to_draw_on_this_line], 
                                    font=font, fill=text_color_rgb)
                
                temp_chars_typed_count += len(line_content)
                if temp_chars_typed_count >= chars_typed_so_far:
                    break # Stop processing lines if all typed characters for this frame are drawn

            # Resize the rendered text frame to original input dimensions
            final_frame_pil = current_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_frame_tensors.append(self.pil_to_tensor_frame(final_frame_pil))

            if chars_typed_so_far >= total_chars_to_type:
                break # Animation complete
            
            chars_typed_so_far += chars_per_frame
            if chars_typed_so_far > total_chars_to_type and (chars_typed_so_far - chars_per_frame) < total_chars_to_type :
                chars_typed_so_far = total_chars_to_type # Ensure the last frame shows everything

        # 5. Batch and Return
        if not output_frame_tensors: # Should not happen with the new loop logic, but as a fallback
            fallback_pil = Image.new("RGB", (original_width, original_height), bg_color_rgb)
            output_frame_tensors.append(self.pil_to_tensor_frame(fallback_pil))
            
        batched_output_tensor = torch.stack(output_frame_tensors, dim=0) # (num_frames, H, W, C)
        return (batched_output_tensor,)

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing ASCIIAnimationGeneratorNode locally...")
    node = ASCIIAnimationGeneratorNode()
    
    # Create a dummy input PIL image for testing
    test_input_pil = Image.new('RGB', (100, 75), color = 'darkcyan') # W, H
    # Convert to ComfyUI-like tensor (B, H, W, C)
    test_input_np = np.array(test_input_pil).astype(np.float32) / 255.0
    dummy_tensor_bhwc = torch.from_numpy(test_input_np).unsqueeze(0)

    print(f"Input tensor shape for test: {dummy_tensor_bhwc.shape}")

    try:
        output_batched_tensor_tuple = node.generate_ascii_animation(
            image=dummy_tensor_bhwc,
            char_width=40, # Smaller for quicker test
            font_path="cour.ttf", 
            font_size=10,
            ascii_charset=" .:oO0@",
            background_color="#202020",
            text_color="#33FF33",
            invert_brightness_mapping=False,
            chars_per_frame=5 # Type 5 characters per frame
        )
        output_batched_tensor = output_batched_tensor_tuple[0]
        print(f"Output batched tensor shape: {output_batched_tensor.shape}") # (num_frames, H, W, C)
        
        num_frames = output_batched_tensor.shape[0]
        print(f"Generated {num_frames} frames.")

        # To save a few frames for visual inspection (e.g., first, middle, last):
        if num_frames > 0:
            indices_to_save = [0]
            if num_frames > 1: indices_to_save.append(num_frames // 2)
            if num_frames > 2: indices_to_save.append(num_frames - 1)
            indices_to_save = sorted(list(set(indices_to_save))) # Unique sorted

            for i, frame_idx in enumerate(indices_to_save):
                if frame_idx < num_frames:
                    frame_tensor_hwc = output_batched_tensor[frame_idx] # H, W, C
                    # Need a tensor_to_pil that handles HWC directly for saving
                    frame_pil = node.tensor_to_pil(frame_tensor_hwc) # Re-use existing one
                    frame_pil.save(f"test_animation_frame_{i+1}_idx{frame_idx}.png")
                    print(f"Saved test_animation_frame_{i+1}_idx{frame_idx}.png")

    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()
