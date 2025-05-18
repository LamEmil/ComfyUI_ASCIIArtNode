import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os

class SequentialTwoPassTypingNode:
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
                "initial_text_color": ("STRING", {"default": "#CCCCCC", "multiline": False}), # Placeholder color
                "invert_brightness_mapping": ("BOOLEAN", {"default": False}),
                "chars_per_frame": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "display":"slider"}),
                "cursor_char": ("STRING", {"default": "_", "multiline": False}),
                "cursor_blink_frames": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1, "display":"slider"}),
                "cursor_color": ("STRING", {"default": "#FFFFFF", "multiline": False}),
                "pause_frames_between_passes": ("INT", {"default": 10, "min": 0, "max": 120, "step":1, "display":"slider"}) # New: Pause
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sequential_two_pass_frames",)
    FUNCTION = "generate_sequential_two_pass_animation"
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
        try:
            bbox = draw.textbbox((0,0), char, font=font)
            return bbox[2] - bbox[0]
        except AttributeError: 
            width, _ = draw.textsize(char, font=font)
            return width

    def generate_sequential_two_pass_animation(
            self, image: torch.Tensor, char_width: int, font_path: str, 
            font_size: int, ascii_charset: str, background_color: str, 
            initial_text_color: str, invert_brightness_mapping: bool, 
            chars_per_frame: int, cursor_char: str, 
            cursor_blink_frames: int, cursor_color: str,
            pause_frames_between_passes: int):
        
        if not ascii_charset: raise ValueError("ASCII charset cannot be empty.")
        if chars_per_frame < 1: chars_per_frame = 1
        if cursor_blink_frames < 1: cursor_blink_frames = 1
        cursor_char = cursor_char[0] if cursor_char else "_"
        if pause_frames_between_passes < 0: pause_frames_between_passes = 0

        if image.ndim == 3: image = image.unsqueeze(0)
        
        input_pil_image_rgb = self.tensor_to_pil(image[0])
        original_width, original_height = input_pil_image_rgb.size

        aspect_ratio = original_height / original_width
        char_height_ascii_grid = max(1, int(char_width * aspect_ratio * 0.5)) 

        # --- Generate the full ASCII map (char_str, image_color, char_width, x_pos, y_pos) ---
        resized_image_for_sampling = input_pil_image_rgb.resize((char_width, char_height_ascii_grid), Image.Resampling.LANCZOS)
        grayscale_image = resized_image_for_sampling.convert("L")
        color_sample_image = resized_image_for_sampling

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Warning: Font '{font_path}' not found. Falling back to default PIL font.")
            try: font = ImageFont.load_default(font_size=font_size)
            except AttributeError: font = ImageFont.load_default()
            except TypeError: font = ImageFont.load_default()
        
        temp_draw = ImageDraw.Draw(Image.new("RGB", (1,1)))
        try: 
            metrics_bbox = temp_draw.textbbox((0,0), "My", font=font) 
            line_pixel_height = metrics_bbox[3] - metrics_bbox[1]
        except AttributeError:
            _, line_pixel_height = temp_draw.textsize("My", font=font)
        if line_pixel_height == 0: line_pixel_height = font_size

        flat_full_ascii_map = [] 
        max_text_pixel_width = 0
        gray_pixels = grayscale_image.load()
        color_pixels = color_sample_image.load()

        for y_idx in range(char_height_ascii_grid):
            current_line_width_pixels = 0
            y_pos_pixels = y_idx * line_pixel_height
            for x_idx in range(char_width):
                brightness = gray_pixels[x_idx, y_idx]
                if invert_brightness_mapping: brightness = 255 - brightness
                char_index = int((brightness / 255) * (len(ascii_charset) - 1))
                char_to_draw_str = ascii_charset[char_index]
                sampled_color_tuple = color_pixels[x_idx, y_idx]
                char_pixel_w = self.get_char_pixel_width(temp_draw, char_to_draw_str, font)
                if char_pixel_w == 0 and char_to_draw_str != ' ':
                    char_pixel_w = self.get_char_pixel_width(temp_draw, "M", font) // 2 
                flat_full_ascii_map.append(
                    (char_to_draw_str, sampled_color_tuple, char_pixel_w, current_line_width_pixels, y_pos_pixels)
                )
                current_line_width_pixels += char_pixel_w
            max_text_pixel_width = max(max_text_pixel_width, current_line_width_pixels)

        if not flat_full_ascii_map:
             default_char_w = self.get_char_pixel_width(temp_draw, " ", font)
             flat_full_ascii_map = [(' ', (128,128,128), default_char_w, 0, 0)]
             max_text_pixel_width = default_char_w
        
        total_chars_in_map = len(flat_full_ascii_map)
        text_render_width = max(1, max_text_pixel_width)
        text_render_height = max(1, char_height_ascii_grid * line_pixel_height)

        try: bg_color_rgb = ImageColor.getrgb(background_color)
        except ValueError: bg_color_rgb = (0,0,0)
        try: initial_text_color_rgb = ImageColor.getrgb(initial_text_color)
        except ValueError: initial_text_color_rgb = (204,204,204) # Default to light gray
        try: cursor_color_rgb = ImageColor.getrgb(cursor_color)
        except ValueError: cursor_color_rgb = (255,255,255)

        output_frame_tensors = []
        animation_frame_index = 0

        # --- Phase 1: Type all characters in initial_text_color ---
        chars_typed_pass1 = 0
        while chars_typed_pass1 <= total_chars_in_map:
            current_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_frame = ImageDraw.Draw(current_frame_pil)
            
            cursor_x_pixel, cursor_y_pixel = 0, 0
            
            for char_idx in range(min(chars_typed_pass1, total_chars_in_map)):
                char_str, _, char_w, x_pos, y_pos = flat_full_ascii_map[char_idx]
                draw_frame.text((x_pos, y_pos), char_str, font=font, fill=initial_text_color_rgb)
                if char_idx == chars_typed_pass1 -1: # Last char drawn in this step
                     cursor_x_pixel = x_pos + char_w
                     cursor_y_pixel = y_pos
            
            if chars_typed_pass1 == 0 and total_chars_in_map > 0: # Cursor at start
                cursor_x_pixel, cursor_y_pixel = 0, flat_full_ascii_map[0][4]


            # Draw cursor for Pass 1
            if chars_typed_pass1 < total_chars_in_map:
                is_cursor_blink_on = (animation_frame_index // cursor_blink_frames) % 2 == 0
                if is_cursor_blink_on:
                    draw_frame.text((cursor_x_pixel, cursor_y_pixel), cursor_char, font=font, fill=cursor_color_rgb)
            
            final_frame_pil = current_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_frame_tensors.append(self.pil_to_tensor_frame(final_frame_pil))
            animation_frame_index += 1

            if chars_typed_pass1 >= total_chars_in_map:
                # Add one last frame for pass 1 if cursor was on, to show text without cursor before pause
                if chars_typed_pass1 == total_chars_in_map and \
                   (animation_frame_index -1 // cursor_blink_frames) % 2 == 0 : # Check if cursor was on previous frame
                    current_frame_pil_no_cursor = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
                    draw_final_p1 = ImageDraw.Draw(current_frame_pil_no_cursor)
                    for char_str_f, _, _, x_pos_f, y_pos_f in flat_full_ascii_map:
                        draw_final_p1.text((x_pos_f, y_pos_f), char_str_f, font=font, fill=initial_text_color_rgb)
                    final_pil_no_cursor = current_frame_pil_no_cursor.resize((original_width, original_height), Image.Resampling.LANCZOS)
                    output_frame_tensors.append(self.pil_to_tensor_frame(final_pil_no_cursor))
                    animation_frame_index +=1
                break
            chars_typed_pass1 = min(total_chars_in_map, chars_typed_pass1 + chars_per_frame)

        # --- Pause Frames ---
        if pause_frames_between_passes > 0 and output_frame_tensors:
            last_pass1_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_pause = ImageDraw.Draw(last_pass1_frame_pil)
            for char_str, _, _, x_pos, y_pos in flat_full_ascii_map:
                draw_pause.text((x_pos, y_pos), char_str, font=font, fill=initial_text_color_rgb)
            final_pause_frame_pil = last_pass1_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            pause_frame_tensor = self.pil_to_tensor_frame(final_pause_frame_pil)
            for _ in range(pause_frames_between_passes):
                output_frame_tensors.append(pause_frame_tensor)
                animation_frame_index += 1
        
        # --- Phase 2: "Color in" characters with image_color ---
        chars_colored_pass2 = 0
        while chars_colored_pass2 <= total_chars_in_map:
            current_frame_pil = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
            draw_frame = ImageDraw.Draw(current_frame_pil)
            
            cursor_x_pixel, cursor_y_pixel = 0, 0

            for char_idx in range(total_chars_in_map):
                char_str, img_color, char_w, x_pos, y_pos = flat_full_ascii_map[char_idx]
                
                current_char_color_p2 = initial_text_color_rgb # Default to initial color
                if char_idx < chars_colored_pass2: # This char has been "colored in"
                    current_char_color_p2 = img_color
                
                draw_frame.text((x_pos, y_pos), char_str, font=font, fill=current_char_color_p2)

                if char_idx == chars_colored_pass2 -1 and chars_colored_pass2 > 0: # Cursor position for pass 2
                     cursor_x_pixel = x_pos + char_w
                     cursor_y_pixel = y_pos

            if chars_colored_pass2 == 0 and total_chars_in_map > 0: # Cursor at start for pass 2
                cursor_x_pixel, cursor_y_pixel = 0, flat_full_ascii_map[0][4]


            # Draw cursor for Pass 2
            if chars_colored_pass2 < total_chars_in_map:
                is_cursor_blink_on = (animation_frame_index // cursor_blink_frames) % 2 == 0
                if is_cursor_blink_on:
                    draw_frame.text((cursor_x_pixel, cursor_y_pixel), cursor_char, font=font, fill=cursor_color_rgb)

            final_frame_pil = current_frame_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            output_frame_tensors.append(self.pil_to_tensor_frame(final_frame_pil))
            animation_frame_index += 1

            if chars_colored_pass2 >= total_chars_in_map:
                # Add one last frame for pass 2 if cursor was on
                if chars_colored_pass2 == total_chars_in_map and \
                   (animation_frame_index -1 // cursor_blink_frames) % 2 == 0 :
                    current_frame_pil_no_cursor = Image.new("RGB", (text_render_width, text_render_height), bg_color_rgb)
                    draw_final_p2 = ImageDraw.Draw(current_frame_pil_no_cursor)
                    for char_str_f, img_color_f, _, x_pos_f, y_pos_f in flat_full_ascii_map:
                        draw_final_p2.text((x_pos_f, y_pos_f), char_str_f, font=font, fill=img_color_f)
                    final_pil_no_cursor = current_frame_pil_no_cursor.resize((original_width, original_height), Image.Resampling.LANCZOS)
                    output_frame_tensors.append(self.pil_to_tensor_frame(final_pil_no_cursor))
                break
            chars_colored_pass2 = min(total_chars_in_map, chars_colored_pass2 + chars_per_frame)

        if not output_frame_tensors:
            fallback_pil = Image.new("RGB", (original_width, original_height), bg_color_rgb)
            output_frame_tensors.append(self.pil_to_tensor_frame(fallback_pil))
            
        batched_output_tensor = torch.stack(output_frame_tensors, dim=0)
        return (batched_output_tensor,)

# For testing the node independently (optional)
if __name__ == '__main__':
    print("Testing SequentialTwoPassTypingNode locally...")
    node = SequentialTwoPassTypingNode()
    
    test_w, test_h = 80, 60 # Small for faster test
    gradient_img = Image.new("RGB", (test_w, test_h))
    gradient_draw = ImageDraw.Draw(gradient_img)
    for i in range(test_w):
        r = int((i / test_w) * 255)
        for j in range(test_h):
            g = int((j / test_h) * 255)
            b = int(((i+j)/(test_w+test_h)) * 180) + 40 
            gradient_draw.point((i,j), fill=(r,g,b))
    
    test_input_np = np.array(gradient_img).astype(np.float32) / 255.0
    dummy_tensor_bhwc = torch.from_numpy(test_input_np).unsqueeze(0)
    print(f"Input tensor shape for test: {dummy_tensor_bhwc.shape}")

    try:
        output_batched_tensor_tuple = node.generate_sequential_two_pass_animation(
            image=dummy_tensor_bhwc,
            char_width=30, 
            font_path="cour.ttf", 
            font_size=10,
            ascii_charset=" ABCDEFGHIJKLMNOPQRSTUVWXYZ.:*#%@",
            background_color="#050505",
            initial_text_color="#999999",
            invert_brightness_mapping=False,
            chars_per_frame=3, 
            cursor_char="|",
            cursor_blink_frames=3,
            cursor_color="#00DD00",
            pause_frames_between_passes=5
        )
        output_batched_tensor = output_batched_tensor_tuple[0]
        print(f"Output batched tensor shape: {output_batched_tensor.shape}")
        
        num_frames = output_batched_tensor.shape[0]
        print(f"Generated {num_frames} frames.")

        save_dir = "test_sequential_two_pass_frames"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving frames to ./{save_dir}/")
        
        indices_to_save = []
        if num_frames > 0:
            indices_to_save.extend(range(min(10, num_frames))) 
            if num_frames > 20:
                indices_to_save.extend(range(num_frames // 2 - 3, num_frames // 2 + 4))
            indices_to_save.extend(range(max(0, num_frames - 10), num_frames)) 
        indices_to_save = sorted(list(set(i for i in indices_to_save if 0 <= i < num_frames)))

        for frame_idx in indices_to_save:
            frame_tensor_hwc = output_batched_tensor[frame_idx]
            frame_np = (frame_tensor_hwc.cpu().numpy() * 255).astype(np.uint8)
            frame_pil = Image.fromarray(frame_np, 'RGB')
            frame_pil.save(os.path.join(save_dir, f"seq_two_pass_frame_{frame_idx:04d}.png"))
        if indices_to_save:
             print(f"Saved selected frames to '{save_dir}' directory. Check for two distinct typing phases.")

    except Exception as e:
        print(f"Error during local test: {e}")
        import traceback
        traceback.print_exc()
