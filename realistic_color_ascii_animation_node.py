import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class RealisticColorASCIIAnimationNode:
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
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}),
                "chars_per_frame": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "display":"slider"}),
                "cursor_char": ("STRING", {"default": "_", "multiline": False}),
                "cursor_blink_frames": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1, "display":"slider"}), # Frames per blink state
                "cursor_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("realistic_animated_ascii_frames",)
    FUNCTION = "generate_realistic_animation"
    CATEGORY = "image/animation" 

    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        img_np = tensor_image.cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            return Image.fromarray(img_np.squeeze(-1), 'L').convert('RGB')
        else:
            return Image.fromarray(img_np, 'RGB' if img_np.shape[-1] == 3 else 'RGBA').convert('RGB')

    def pil_to_tensor_frame(self, pil_image: Image.Image) -> torch.Tensor:
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        if img_np.ndim == 2: 
            img_np = np.expand_dims(img_np, axis=2)
            img_np = np.repeat(img_np, 3, axis=2)
        return torch.from_numpy(img_np) 

    def get_char_pixel_width(self, draw, char, font):
        """Helper to get character width, robust to Pillow versions."""
        try:
            bbox = draw.textbbox((0,0), char, font=font)
            return bbox[2] - bbox[0]
        except AttributeError: # Fallback for older Pillow
            width, _ = draw.textsize(char, font=font)
            return width

    def generate_realistic_animation(self, image: torch.Tensor, char_width: int, font_path: str, 
                                       font_size: int, ascii_charset: str, background_color: str, 
                                       invert_brightness_mapping: bool, chars_per_frame: int,
                                       cursor_char: str, cursor_blink_frames: int, cursor_color: str):
        
        if not ascii_charset: raise ValueError("ASCII charset cannot be empty.")
        if chars_per_frame < 1: chars_per_frame = 1
        if cursor_blink_frames < 1: cursor_blink_frames = 1
        cursor_char = cursor_char[0] if cursor_char else "_" # Ensure single character

        if image.ndim == 3: image = image.unsqueeze(0)
        
        input_pil_image_rgb = self.tensor_to_pil(image[0])
        original_width, original_height = input_pil_image_rgb.size

        aspect_ratio = original_height / original_width
        char_height_ascii_grid = max(1, int(char_width * aspect_ratio * 0.5)) 

        resized_image_for_sampling = input_pil_image_rgb.resize((char_width, char_height_ascii_grid), Image.Resampling.LANCZOS)
        grayscale_image = resized_image_for_sampling.convert("L")
        color_sample_image = resized_image_for_sampling # Already RGB

        full_ascii_map = [] # Stores list of lines, each line is list of (char_str, (r,g,b), char_pixel_width)
        
        gray_pixels = grayscale_image.load()
        color_pixels = color_sample_image.load()

        # Pre-calculate font and character dimensions for layout
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try: font = ImageFont.load_default(font_size=font_size)
            except AttributeError: font = ImageFont.load_default()
            except TypeError: font = ImageFont.load_default()
        
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
        try: 
            # Use a character with ascenders/descenders for better line height
            metrics_bbox = temp_draw.textbbox((0,0), "My", font=font) 
            line_pixel_height = metrics_bbox[3] - metrics_bbox[1]
        except AttributeError:
            _, line_pixel_height = temp_draw.textsize("My", font=font)
        if line_pixel_height == 0: line_pixel_height = font_size # Fallback

        max_text_pixel_width = 0
        for y_idx in range(char_height_ascii_grid):
            line_map = []
            current_line_width = 0
            for x_idx in range(char_width):
                brightness = gray_pixels[x_idx, y_idx]
                if invert_brightness_mapping: brightness = 255 - brightness
                
                char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                char_to_draw_str = ascii_charset[char_index]
                sampled_color_tuple = color_pixels[x_idx, y_idx]
                
                char_pixel_w = self.get_char_pixel_width(temp_draw, char_to_draw_str, font)
                if char_pixel_w == 0 and char_to_draw_str != ' ': # Avoid zero width for non-space
                    char_pixel_w = self.get_char_pixel_width(temp_draw, "M", font) // 2 # Fallback
                
                line_map.append((char_to_draw_str, sampled_color_tuple, char_pixel_w))
                current_line_width += char_pixel_w
            full_ascii_map.append(line_map)
            max_text_pixel_width = max(max_text_pixel_width, current_line_width)

        if not any(full_ascii_map) or not any(line for line in full_ascii_map):
             default_char_w = self.get_char_pixel_width(temp_draw, " ", font)
             full_ascii_map = [[(' ', (128,128,128), default_char_w)]]
             max_text_pixel_width = default_char_w

        text_render_width = max(1, max_text_pixel_width)
        text_render_height = max(1, len(full_ascii_map) * line_pixel_height)

        try: bg_color_rgb = ImageColor.getrgb(background_color)
        except ValueError: bg_color_rgb = (0,0,0)
        try: cursor_color_rgb = ImageColor.getrgb(cursor_color)
        except ValueError: cursor_color_rgb = (255,255,255)

        output_frame_tensors = []
        total_chars_in_map = sum(len(line) for line in full_ascii_map)
        chars_typed_count = 0
        animation_frame_index = 0
        
        while chars_typed_count <= total_chars_in_map:
            current_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_frame = ImageDraw.Draw(current_frame_pil)
            
            # --- Draw typed characters ---
            temp_chars_drawn_this_frame = 0
            cursor_x_pos_pixels = 0
            cursor_y_pos_pixels = 0
            found_cursor_pos = False

            for line_idx, line_content_map in enumerate(full_ascii_map):
                current_x_offset_pixels = 0
                for char_map_idx, (char_str, char_rgb, char_w_pixels) in enumerate(line_content_map):
                    if temp_chars_drawn_this_frame < chars_typed_count:
                        draw_frame.text((current_x_offset_pixels, line_idx * line_pixel_height), 
                                        char_str, font=font, fill=char_rgb)
                        temp_chars_drawn_this_frame += 1
                    
                    # Determine cursor position (immediately after the last typed char, or start of line if no chars typed yet on this line)
                    if not found_cursor_pos and temp_chars_drawn_this_frame == chars_typed_count:
                        cursor_x_pos_pixels = current_x_offset_pixels
                        cursor_y_pos_pixels = line_idx * line_pixel_height
                        found_cursor_pos = True # Cursor will be drawn here
                    
                    current_x_offset_pixels += char_w_pixels
                
                if found_cursor_pos and chars_typed_count > 0 and temp_chars_drawn_this_frame == chars_typed_count and chars_typed_count == sum(len(l) for l in full_ascii_map[:line_idx+1]):
                    # If cursor is at the end of a line (and it's not the very first char of all text)
                    # and it's not the last char overall, move to start of next line if exists
                    if line_idx + 1 < len(full_ascii_map):
                        cursor_x_pos_pixels = 0
                        cursor_y_pos_pixels = (line_idx + 1) * line_pixel_height
                    # else it stays at end of current line if it's the very last char overall
                    
            # If all characters are typed, cursor_pos might not be set if loop finishes exactly.
            # Or if chars_typed_count is 0, cursor is at the beginning.
            if not found_cursor_pos and chars_typed_count == 0:
                cursor_x_pos_pixels = 0
                cursor_y_pos_pixels = 0
                found_cursor_pos = True

            # --- Draw flashing cursor ---
            if found_cursor_pos and chars_typed_count < total_chars_in_map: # Only draw cursor if not all chars are typed
                is_cursor_blink_on = (animation_frame_index // cursor_blink_frames) % 2 == 0
                if is_cursor_blink_on:
                    draw_frame.text((cursor_x_pos_pixels, cursor_y_pos_pixels),
                                    cursor_char, font=font, fill=cursor_color_rgb)
            
            final_frame_pil = current_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_frame_tensors.append(self.pil_to_tensor_frame(final_frame_pil))

            if chars_typed_count >= total_chars_in_map:
                # If all characters are typed, generate one last frame without cursor (if it was on)
                # or if it was already off, this is the final state.
                if is_cursor_blink_on and found_cursor_pos: # Check if cursor was drawn
                    current_frame_pil_no_cursor = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
                    draw_final = ImageDraw.Draw(current_frame_pil_no_cursor)
                    # Redraw text without cursor
                    temp_c = 0
                    for l_idx, l_map in enumerate(full_ascii_map):
                        curr_x = 0
                        for ch_str, ch_rgb, ch_w in l_map:
                            if temp_c < chars_typed_count: # Should be all chars now
                                draw_final.text((curr_x, l_idx * line_pixel_height), ch_str, font=font, fill=ch_rgb)
                            curr_x += ch_w
                            temp_c +=1
                    final_pil_no_cursor = current_frame_pil_no_cursor.resize((original_width, original_height), Image.Resampling.LANCZOS)
                    output_frame_tensors.append(self.pil_to_tensor_frame(final_pil_no_cursor))
                break 
            
            chars_typed_count += chars_per_frame
            # Ensure the last step types exactly the remaining characters
            if chars_typed_count > total_chars_in_map and (chars_typed_count - chars_per_frame) < total_chars_in_map :
                chars_typed_count = total_chars_in_map
            
            animation_frame_index += 1

        if not output_frame_tensors:
            fallback_pil = Image.new("RGB", (original_width, original_height), bg_color_rgb)
            output_frame_tensors.append(self.pil_to_tensor_frame(fallback_pil))
            
        batched_output_tensor = torch.stack(output_frame_tensors, dim=0)
        return (batched_output_tensor,)

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing RealisticColorASCIIAnimationNode locally...")
    node = RealisticColorASCIIAnimationNode()
    
    test_w, test_h = 100, 70
    gradient_img = Image.new("RGB", (test_w, test_h))
    gradient_draw = ImageDraw.Draw(gradient_img)
    for i in range(test_w):
        r = int((i / test_w) * 255)
        for j in range(test_h):
            g = int((j / test_h) * 255)
            b = int(((i+j)/(test_w+test_h)) * 255)
            gradient_draw.point((i,j), fill=(r,g,b))
    
    test_input_np = np.array(gradient_img).astype(np.float32) / 255.0
    dummy_tensor_bhwc = torch.from_numpy(test_input_np).unsqueeze(0)

    print(f"Input tensor shape for test: {dummy_tensor_bhwc.shape}")

    try:
        output_batched_tensor_tuple = node.generate_realistic_animation(
            image=dummy_tensor_bhwc,
            char_width=30, 
            font_path="cour.ttf", 
            font_size=12,
            ascii_charset=" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:-=+*#%@",
            background_color="#050505",
            invert_brightness_mapping=False,
            chars_per_frame=2,
            cursor_char="|",
            cursor_blink_frames=3,
            cursor_color="#00FF00"
        )
        output_batched_tensor = output_batched_tensor_tuple[0]
        print(f"Output batched tensor shape: {output_batched_tensor.shape}")
        
        num_frames = output_batched_tensor.shape[0]
        print(f"Generated {num_frames} frames.")

        # Save some frames
        save_dir = "test_realistic_frames"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving frames to ./{save_dir}/")
        
        # Determine indices to save (e.g., first 5, some middle, last 5)
        indices_to_save = []
        if num_frames > 0:
            indices_to_save.extend(range(min(5, num_frames))) # First 5
            if num_frames > 10:
                indices_to_save.append(num_frames // 2 -1)
                indices_to_save.append(num_frames // 2)
                indices_to_save.append(num_frames // 2 +1)
            indices_to_save.extend(range(max(0, num_frames - 5), num_frames)) # Last 5
        indices_to_save = sorted(list(set(i for i in indices_to_save if 0 <= i < num_frames)))


        for frame_idx in indices_to_save:
            frame_tensor_hwc = output_batched_tensor[frame_idx]
            frame_np = (frame_tensor_hwc.cpu().numpy() * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np, 'RGB')
            frame_pil.save(os.path.join(save_dir, f"realistic_frame_{frame_idx:03d}.png"))
        if indices_to_save:
             print(f"Saved selected frames to '{save_dir}' directory.")


    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()

