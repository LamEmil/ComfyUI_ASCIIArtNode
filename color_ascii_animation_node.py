import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class ColorASCIIAnimationGeneratorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        default_charset = " .'`^\":;Il!i~_?[{1(|/fjxnvzXYJCLQ0Zmwqdkh*#M&8%B@$"
        
        return {
            "required": {
                "image": ("IMAGE",),
                "char_width": ("INT", {"default": 80, "min": 10, "max": 1000, "step": 10, "display": "slider"}),
                "font_path": ("STRING", {"default": "cour.ttf", "multiline": False}), 
                "font_size": ("INT", {"default": 15, "min": 5, "max": 100, "step": 1, "display": "slider"}),
                "ascii_charset": ("STRING", {"default": default_charset, "multiline": True}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                # text_color is removed as character color comes from the image
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}),
                "chars_per_frame": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display":"slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",) # Output is a batch of images
    RETURN_NAMES = ("colored_animated_ascii_frames",)
    FUNCTION = "generate_color_ascii_animation"
    CATEGORY = "image/animation" 

    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        """Converts a HWC PyTorch tensor (0-1 float) from the input batch to a PIL Image."""
        img_np = tensor_image.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            return Image.fromarray(img_np.squeeze(-1), 'L').convert('RGB') # Ensure RGB for color sampling
        else:
            return Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA').convert('RGB')


    def pil_to_tensor_frame(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a HWC PyTorch tensor (0-1 float) for a single animation frame."""
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0 # Ensure RGB
        if img_np.ndim == 2: 
            img_np = np.expand_dims(img_np, axis=2)
            img_np = np.repeat(img_np, 3, axis=2) # Convert grayscale to RGB by repeating channel
        
        tensor_image = torch.from_numpy(img_np) 
        return tensor_image

    def generate_color_ascii_animation(self, image: torch.Tensor, char_width: int, font_path: str, 
                                       font_size: int, ascii_charset: str, background_color: str, 
                                       invert_brightness_mapping: bool, chars_per_frame: int):
        
        if not ascii_charset:
            raise ValueError("ASCII charset cannot be empty.")
        if chars_per_frame < 1:
            chars_per_frame = 1

        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        input_pil_image_rgb = self.tensor_to_pil(image[0]) # Ensure it's RGB
        original_width, original_height = input_pil_image_rgb.size

        aspect_ratio = original_height / original_width
        char_height = max(1, int(char_width * aspect_ratio * 0.5)) 

        # Image for brightness mapping (grayscale)
        resized_image_for_brightness = input_pil_image_rgb.resize((char_width, char_height), Image.Resampling.LANCZOS)
        grayscale_image = resized_image_for_brightness.convert("L")
        
        # Image for color sampling (RGB, same dimensions as grayscale)
        # This ensures direct correspondence between character position and color sample
        color_sample_image = resized_image_for_brightness # Already RGB and resized

        full_ascii_map = [] # Will store list of lines, where each line is list of (char, (r,g,b))
        
        gray_pixels = grayscale_image.load()
        color_pixels = color_sample_image.load()

        for y_idx in range(char_height):
            line_map = []
            for x_idx in range(char_width):
                brightness = gray_pixels[x_idx, y_idx]
                if invert_brightness_mapping:
                    brightness = 255 - brightness
                
                char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                char_to_draw = ascii_charset[char_index]
                
                # Sample color from the color_sample_image at the same (x,y)
                sampled_color = color_pixels[x_idx, y_idx] # This will be an (R, G, B) tuple
                line_map.append((char_to_draw, sampled_color))
            full_ascii_map.append(line_map)

        if not any(full_ascii_map):
             # Ensure at least one space with a default color if map is empty
            full_ascii_map = [[(' ', (128,128,128))]]


        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try: font = ImageFont.load_default(font_size=font_size)
            except AttributeError: font = ImageFont.load_default()
            except TypeError: font = ImageFont.load_default()
        
        try:
            bg_color_rgb = ImageColor.getrgb(background_color)
        except ValueError:
            print(f"Warning: Invalid background color. Using black.")
            bg_color_rgb = (0,0,0)

        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
        max_text_pixel_width = 0
        
        # Estimate max width by rendering a line of 'M's (a wide character)
        # This is a simplification; true max width depends on the actual characters.
        # A more accurate way would be to iterate through full_ascii_map and sum widths,
        # but that's more complex if characters are not monospaced.
        # For monospaced fonts, this is simpler: char_width * (width of one char).
        
        # Using textbbox for a single character to estimate width/height
        try: 
            char_bbox = temp_draw.textbbox((0,0), "M", font=font) # 'M' is often a wide char
            single_char_width = char_bbox[2] - char_bbox[0]
            line_pixel_height = char_bbox[3] - char_bbox[1]
        except AttributeError: # Fallback for older Pillow
            single_char_width, line_pixel_height = temp_draw.textsize("M", font=font)

        if single_char_width == 0: single_char_width = font_size // 2 # Rough fallback
        if line_pixel_height == 0: line_pixel_height = font_size # Rough fallback

        max_text_pixel_width = char_width * single_char_width
        
        text_render_width = max(1, max_text_pixel_width)
        text_render_height = max(1, len(full_ascii_map) * line_pixel_height)

        output_frame_tensors = []
        total_chars_to_type = sum(len(line) for line in full_ascii_map)
        chars_typed_so_far = 0
        
        while chars_typed_so_far <= total_chars_to_type:
            current_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_frame = ImageDraw.Draw(current_frame_pil)
            
            temp_chars_typed_count = 0 

            for line_idx, line_content_map in enumerate(full_ascii_map):
                current_x_offset = 0
                for char_idx, (char_to_draw, char_color) in enumerate(line_content_map):
                    if temp_chars_typed_count < chars_typed_so_far:
                        draw_frame.text((current_x_offset, line_idx * line_pixel_height), 
                                        char_to_draw, 
                                        font=font, 
                                        fill=char_color) # Use individual char_color
                    
                    # Get width of current character to advance x_offset
                    try:
                        bbox = draw_frame.textbbox((0,0), char_to_draw, font=font)
                        char_pixel_width = bbox[2] - bbox[0]
                    except AttributeError:
                        char_pixel_width, _ = draw_frame.textsize(char_to_draw, font=font)
                    
                    current_x_offset += char_pixel_width
                    temp_chars_typed_count += 1

                    if temp_chars_typed_count >= chars_typed_so_far:
                        break # Break from inner loop (chars in line)
                if temp_chars_typed_count >= chars_typed_so_far:
                    break # Break from outer loop (lines)
            
            final_frame_pil = current_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_frame_tensors.append(self.pil_to_tensor_frame(final_frame_pil))

            if chars_typed_so_far >= total_chars_to_type:
                break 
            
            chars_typed_so_far += chars_per_frame
            if chars_typed_so_far > total_chars_to_type and (chars_typed_so_far - chars_per_frame) < total_chars_to_type :
                chars_typed_so_far = total_chars_to_type

        if not output_frame_tensors:
            fallback_pil = Image.new("RGB", (original_width, original_height), bg_color_rgb)
            output_frame_tensors.append(self.pil_to_tensor_frame(fallback_pil))
            
        batched_output_tensor = torch.stack(output_frame_tensors, dim=0)
        return (batched_output_tensor,)

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing ColorASCIIAnimationGeneratorNode locally...")
    node = ColorASCIIAnimationGeneratorNode()
    
    # Create a dummy input PIL image with varied colors
    test_w, test_h = 120, 90
    gradient_img = Image.new("RGB", (test_w, test_h))
    gradient_draw = ImageDraw.Draw(gradient_img)
    for i in range(test_w):
        r = int((i / test_w) * 255)
        for j in range(test_h):
            g = int((j / test_h) * 255)
            b = 128
            gradient_draw.point((i,j), fill=(r,g,b))
    
    test_input_np = np.array(gradient_img).astype(np.float32) / 255.0
    dummy_tensor_bhwc = torch.from_numpy(test_input_np).unsqueeze(0)

    print(f"Input tensor shape for test: {dummy_tensor_bhwc.shape}")

    try:
        output_batched_tensor_tuple = node.generate_color_ascii_animation(
            image=dummy_tensor_bhwc,
            char_width=60, 
            font_path="cour.ttf", 
            font_size=12,
            ascii_charset=" .:oO0@",
            background_color="#111111",
            invert_brightness_mapping=False,
            chars_per_frame=10 
        )
        output_batched_tensor = output_batched_tensor_tuple[0]
        print(f"Output batched tensor shape: {output_batched_tensor.shape}")
        
        num_frames = output_batched_tensor.shape[0]
        print(f"Generated {num_frames} frames.")

        if num_frames > 0:
            indices_to_save = [0]
            if num_frames > 1: indices_to_save.append(num_frames // 2)
            if num_frames > 2: indices_to_save.append(num_frames - 1)
            indices_to_save = sorted(list(set(indices_to_save)))

            for i, frame_idx in enumerate(indices_to_save):
                if frame_idx < num_frames:
                    frame_tensor_hwc = output_batched_tensor[frame_idx]
                    # Need a tensor_to_pil that handles HWC directly for saving
                    # The existing tensor_to_pil in the class expects a batched tensor's HWC slice
                    # So we make a dummy batch for it or adapt
                    pil_converter = node.tensor_to_pil # This expects HWC from image[0]
                    
                    # For saving, we need to convert HWC tensor to PIL
                    frame_np = (frame_tensor_hwc.cpu().numpy() * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np, 'RGB')
                    
                    frame_pil.save(f"test_color_animation_frame_{i+1}_idx{frame_idx}.png")
                    print(f"Saved test_color_animation_frame_{i+1}_idx{frame_idx}.png")

    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()
