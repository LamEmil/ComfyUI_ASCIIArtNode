import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class VideoToDynamicColorASCIIArtNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        default_charset = " .'`^\":;Il!i~_?[{1(|/fjxnvzXYJCLQ0Zmwqdkh*#M&8%B@$" 
        return {
            "required": {
                "video_frames": ("IMAGE",), # Expecting batched images (B, H, W, C)
                "char_width": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10, "display": "slider"}),
                "font_path": ("STRING", {"default": "cour.ttf", "multiline": False}), 
                "font_size": ("INT", {"default": 15, "min": 5, "max": 100, "step": 1, "display": "slider"}),
                "ascii_charset": ("STRING", {"default": default_charset, "multiline": True}),
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}), # For char selection per frame
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("dynamic_ascii_video_frames",)
    FUNCTION = "generate_video_to_dynamic_color_ascii"
    CATEGORY = "video/art" 

    def tensor_to_pil(self, tensor_frame: torch.Tensor) -> Image.Image:
        """Converts a single HWC PyTorch tensor frame (0-1 float) to a PIL Image."""
        img_np = tensor_frame.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1: # Grayscale
            return Image.fromarray(img_np.squeeze(-1), 'L').convert('RGB') # Ensure RGB
        else: # RGB or RGBA
            return Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA').convert('RGB')

    def pil_to_tensor_frame(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a HWC PyTorch tensor (0-1 float) for an output frame."""
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        if img_np.ndim == 2: 
            img_np = np.expand_dims(img_np, axis=2)
            img_np = np.repeat(img_np, 3, axis=2)
        return torch.from_numpy(img_np) # HWC

    def get_char_dimensions(self, font):
        """Estimates character width and height for a monospaced font."""
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
        try: 
            bbox_m = temp_draw.textbbox((0,0), "M", font=font)
            char_w = bbox_m[2] - bbox_m[0]
            bbox_my = temp_draw.textbbox((0,0), "My", font=font)
            char_h = bbox_my[3] - bbox_my[1]
        except AttributeError: 
            char_w, _ = temp_draw.textsize("M", font=font)
            _, char_h = temp_draw.textsize("My", font=font)
        
        if char_w == 0: char_w = font.size // 2 if hasattr(font, 'size') else 10 
        if char_h == 0: char_h = font.size if hasattr(font, 'size') else 12    
        return char_w, char_h

    def generate_video_to_dynamic_color_ascii(self, video_frames: torch.Tensor, char_width: int, font_path: str, 
                                              font_size: int, ascii_charset: str, background_color: str, 
                                              invert_brightness_mapping: bool):
        
        if not ascii_charset: raise ValueError("ASCII charset cannot be empty.")
        if video_frames is None or video_frames.ndim != 4 or video_frames.shape[0] == 0:
            print("Warning: Input video_frames are invalid or empty. Returning empty tensor.")
            return (torch.empty(0,1,1,3),) 

        num_input_frames, original_height, original_width, _ = video_frames.shape

        # Prepare Font and Background Color (once)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try: font = ImageFont.load_default(font_size=font_size)
            except AttributeError: font = ImageFont.load_default()
            except TypeError: font = ImageFont.load_default()
        
        cell_w, cell_h = self.get_char_dimensions(font)
        
        try:
            bg_color_rgb = ImageColor.getrgb(background_color)
        except ValueError:
            print(f"Warning: Invalid background color string. Using black.")
            bg_color_rgb = (0,0,0)

        output_video_frame_tensors = []

        # Calculate ASCII grid dimensions (once, based on first frame's aspect ratio for consistency if needed,
        # or could be recalculated if char_width/height are meant to adapt to varying input aspect ratios,
        # but for fixed output char_width, this is fine)
        aspect_ratio = original_height / original_width
        char_height_ascii_grid = max(1, int(char_width * aspect_ratio * 0.5))
        
        text_render_width = char_width * cell_w
        text_render_height = char_height_ascii_grid * cell_h

        for i in range(num_input_frames):
            current_input_frame_pil = self.tensor_to_pil(video_frames[i]) # HWC PIL Image
            
            # --- Dynamic ASCII Character Selection for CURRENT frame ---
            resized_current_frame_for_ascii = current_input_frame_pil.resize(
                (char_width, char_height_ascii_grid), Image.Resampling.LANCZOS
            )
            grayscale_current_frame = resized_current_frame_for_ascii.convert("L")
            gray_pixels_current_frame = grayscale_current_frame.load()

            current_frame_char_map = [[' ' for _ in range(char_width)] for _ in range(char_height_ascii_grid)]
            for y_char in range(char_height_ascii_grid):
                for x_char in range(char_width):
                    brightness = gray_pixels_current_frame[x_char, y_char]
                    if invert_brightness_mapping: brightness = 255 - brightness
                    char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                    current_frame_char_map[y_char][x_char] = ascii_charset[char_index]
            
            # --- Color Sampling from CURRENT frame ---
            # The resized_current_frame_for_ascii is already RGB and at the correct dimensions for color sampling
            color_pixels_current_frame = resized_current_frame_for_ascii.load() # This is an RGB image's pixels

            # --- Render the current frame's dynamic ASCII with dynamic colors ---
            rendered_ascii_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_output = ImageDraw.Draw(rendered_ascii_frame_pil)

            for y_char_idx in range(char_height_ascii_grid):
                for x_char_idx in range(char_width):
                    char_to_render = current_frame_char_map[y_char_idx][x_char_idx]
                    sampled_color = color_pixels_current_frame[x_char_idx, y_char_idx]
                    
                    x_pos_pixels = x_char_idx * cell_w
                    y_pos_pixels = y_char_idx * cell_h
                    
                    draw_output.text((x_pos_pixels, y_pos_pixels), char_to_render, font=font, fill=sampled_color)
            
            final_output_frame_pil = rendered_ascii_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_video_frame_tensors.append(self.pil_to_tensor_frame(final_output_frame_pil))

        if not output_video_frame_tensors:
             h_out = original_height if 'original_height' in locals() else 1
             w_out = original_width if 'original_width' in locals() else 1
             return (torch.empty(0, h_out, w_out, 3),)

        batched_output_tensor = torch.stack(output_video_frame_tensors, dim=0)
        return (batched_output_tensor,)

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing VideoToDynamicColorASCIIArtNode locally...")
    node = VideoToDynamicColorASCIIArtNode()

    num_test_frames = 5
    test_h, test_w = 60, 80
    dummy_video_list = []
    for f_idx in range(num_test_frames):
        frame_img = Image.new("RGB", (test_w, test_h))
        frame_draw = ImageDraw.Draw(frame_img)
        # Create a moving gradient effect for testing dynamic char changes
        for i in range(test_w):
            for j in range(test_h):
                r_base = int((i / test_w) * 255)
                g_base = int((j / test_h) * 255)
                # Shift colors and brightness over frames
                r = (r_base + f_idx * 20) % 256
                g = (g_base + f_idx * 10) % 256
                b = (100 + f_idx * 30) % 256
                frame_draw.point((i,j), fill=(r, g, b))
        
        frame_np = np.array(frame_img).astype(np.float32) / 255.0
        dummy_video_list.append(torch.from_numpy(frame_np))
    
    dummy_video_tensor_bhwc = torch.stack(dummy_video_list, dim=0)
    print(f"Input video tensor shape for test: {dummy_video_tensor_bhwc.shape}")

    try:
        output_batched_tensor_tuple = node.generate_video_to_dynamic_color_ascii(
            video_frames=dummy_video_tensor_bhwc,
            char_width=40, 
            font_path="cour.ttf", 
            font_size=10,
            ascii_charset=" .:oO0@#", # Even shorter for more visible changes
            background_color="#101010",
            invert_brightness_mapping=False
        )
        output_batched_tensor = output_batched_tensor_tuple[0]
        print(f"Output batched tensor shape: {output_batched_tensor.shape}")
        
        output_num_frames = output_batched_tensor.shape[0]
        print(f"Generated {output_num_frames} output frames.")

        save_dir = "test_dynamic_video_ascii_frames"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving frames to ./{save_dir}/")
        
        indices_to_save = [i for i in range(min(output_num_frames, 5))]

        for frame_idx in indices_to_save:
            frame_tensor_hwc = output_batched_tensor[frame_idx]
            frame_np = (frame_tensor_hwc.cpu().numpy() * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np, 'RGB')
            frame_pil.save(os.path.join(save_dir, f"dynamic_video_ascii_frame_{frame_idx:03d}.png"))
        if indices_to_save:
             print(f"Saved selected frames to '{save_dir}' directory.")

    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()
