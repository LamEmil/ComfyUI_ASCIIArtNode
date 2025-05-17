import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class ASCIIArtGeneratorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        default_charset = " .'`^\":;Il!i~_?[{1(|/fjxnvzXYJCLQ0Zmwqdkh*#M&8%B@$"
        
        return {
            "required": {
                "image": ("IMAGE",),
                "char_width": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 10, "display": "slider"}),
                "font_path": ("STRING", {"default": "cour.ttf", "multiline": False}), 
                "font_size": ("INT", {"default": 15, "min": 5, "max": 100, "step": 1, "display": "slider"}),
                "ascii_charset": ("STRING", {"default": default_charset, "multiline": True}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "text_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ascii_image",)
    FUNCTION = "generate_ascii_art"
    CATEGORY = "image/art"

    def generate_ascii_art(self, image: torch.Tensor, char_width: int, font_path: str, 
                           font_size: int, ascii_charset: str, background_color: str, 
                           text_color: str, invert_brightness_mapping: bool):
        
        # 0. Basic Input Validation
        if not ascii_charset:
            raise ValueError("ASCII charset cannot be empty.")

        # 1. Convert Tensor to PIL Image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        input_pil_image = self.tensor_to_pil(image[0]) 
        original_width, original_height = input_pil_image.size # Store original dimensions

        # 2. Convert PIL Image to ASCII representation
        aspect_ratio = original_height / original_width
        # Ensure char_height is at least 1, even if aspect_ratio or char_width is small
        char_height = max(1, int(char_width * aspect_ratio * 0.5)) # Adjusted for typical char aspect ratio

        resized_image_for_ascii = input_pil_image.resize((char_width, char_height), Image.Resampling.LANCZOS)
        grayscale_image = resized_image_for_ascii.convert("L")

        ascii_lines = []
        pixels = grayscale_image.load()

        for y_idx in range(char_height):
            line_of_text = ""
            for x_idx in range(char_width):
                brightness = pixels[x_idx, y_idx]
                if invert_brightness_mapping:
                    brightness = 255 - brightness
                
                char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                line_of_text += ascii_charset[char_index]
            ascii_lines.append(line_of_text)

        # 3. Render ASCII Text to a New Image (at its "natural" text size first)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try:
                font = ImageFont.load_default(font_size=font_size)
            except AttributeError: # Older Pillow
                 font = ImageFont.load_default()
            except TypeError: # If font_size is not accepted by load_default
                 font = ImageFont.load_default()


        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1))) 
        
        max_line_pixel_width = 0
        if ascii_lines:
            for line in ascii_lines:
                # Use textbbox for more accurate width, available in newer Pillow versions
                try:
                    line_bbox = temp_draw.textbbox((0,0), line, font=font)
                    max_line_pixel_width = max(max_line_pixel_width, line_bbox[2] - line_bbox[0])
                except AttributeError: # Fallback for older Pillow using textsize
                    line_width, _ = temp_draw.textsize(line, font=font)
                    max_line_pixel_width = max(max_line_pixel_width, line_width)

        # Estimate line height
        try:
            # Using textbbox for a character with ascenders/descenders
            char_metrics_bbox = temp_draw.textbbox((0,0), "My", font=font)
            line_pixel_height = char_metrics_bbox[3] - char_metrics_bbox[1] 
        except AttributeError: # Fallback for older Pillow
            _, line_pixel_height_fallback = temp_draw.textsize("My", font=font)
            line_pixel_height = line_pixel_height_fallback

        if line_pixel_height == 0 and font_size > 0 : 
            line_pixel_height = int(font_size * 1.2) 

        if max_line_pixel_width == 0: max_line_pixel_width = char_width * font_size // 2 
        if line_pixel_height == 0: line_pixel_height = font_size 

        text_render_width = max(1, max_line_pixel_width)
        text_render_height = max(1, len(ascii_lines) * line_pixel_height)

        try:
            bg_color_rgb = ImageColor.getrgb(background_color)
            text_color_rgb = ImageColor.getrgb(text_color)
        except ValueError:
            print(f"Warning: Invalid color string. Using black background and white text.")
            bg_color_rgb = (0,0,0)
            text_color_rgb = (255,255,255)

        # Create the initial rendered image based on text dimensions
        rendered_text_image = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
        draw = ImageDraw.Draw(rendered_text_image)

        current_y = 0
        for line in ascii_lines:
            draw.text((0, current_y), line, font=font, fill=text_color_rgb)
            current_y += line_pixel_height
            
        # 4. Resize the rendered ASCII art to match the original input image dimensions
        # This is the key change to match input image size.
        final_output_image = rendered_text_image.resize((original_width, original_height), Image.Resampling.LANCZOS)
            
        # 5. Convert Final PIL Image back to Tensor
        output_tensor = self.pil_to_tensor(final_output_image)

        return (output_tensor,)

    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        img_np = tensor_image.cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
             img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            return Image.fromarray(img_np.squeeze(-1), 'L')
        else:
            return Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA')

    def pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=2)
        tensor_image = torch.from_numpy(img_np)
        return tensor_image.unsqueeze(0) # Add batch dimension

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing ASCIIArtGeneratorNode locally...")
    node = ASCIIArtGeneratorNode()
    
    # Create a dummy input PIL image for testing
    test_input_pil = Image.new('RGB', (200, 150), color = 'blue') # W, H
    # Convert to tensor (B, H, W, C)
    test_input_np = np.array(test_input_pil).astype(np.float32) / 255.0
    dummy_tensor_bhwc = torch.from_numpy(test_input_np).unsqueeze(0)

    print(f"Input tensor shape for test: {dummy_tensor_bhwc.shape}")

    try:
        output_image_tensor_tuple = node.generate_ascii_art(
            image=dummy_tensor_bhwc,
            char_width=80,
            font_path="cour.ttf", 
            font_size=10,
            ascii_charset=" .:-=+*#%@",
            background_color="#101010",
            text_color="#00FF00",
            invert_brightness_mapping=False
        )
        output_image_tensor = output_image_tensor_tuple[0]
        print(f"Output tensor shape: {output_image_tensor.shape}")
        
        # Check if output dimensions match input dimensions
        # Input was (1, 150, 200, 3) -> H=150, W=200
        # Output should be (1, 150, 200, 3)
        if output_image_tensor.shape[1] == 150 and output_image_tensor.shape[2] == 200:
            print("SUCCESS: Output dimensions match input dimensions.")
        else:
            print(f"FAILURE: Output dimensions {output_image_tensor.shape[1]}x{output_image_tensor.shape[2]} "
                  f"do not match input 150x200.")

        output_pil_image = node.tensor_to_pil(output_image_tensor[0]) 
        output_pil_image.save("test_ascii_output_same_size.png")
        print("Saved test_ascii_output_same_size.png")

    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()
