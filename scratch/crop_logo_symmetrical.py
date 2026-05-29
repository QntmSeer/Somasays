import os
from PIL import Image

pristine_path = r"C:\Users\Gebruiker\.gemini\antigravity\brain\fe99369f-930f-4f8c-9cfd-252c3583cd05\somasays_pixel_logo_slate_teal_1780050859480.png"
repo_logo_path = r"c:\Users\Gebruiker\Documents\Computational Bio\Somasays\somasays_logo.png"

if not os.path.exists(pristine_path):
    print(f"[ERROR] Pristine logo not found at {pristine_path}")
    exit(1)

img = Image.open(pristine_path)
if img.mode not in ("RGB", "RGBA"):
    img = img.convert("RGB")

width, height = img.size
bg_color = img.getpixel((0, 0))

# Noise-tolerant difference checker
def is_different(c1, c2, tolerance=20):
    return sum(abs(a - b) for a, b in zip(c1[:3], c2[:3])) > tolerance

# Find vertical boundaries with noise filtering (require at least 3 non-bg pixels in a row)
first_y = None
last_y = None

for y in range(height):
    different_pixels_in_row = 0
    for x in range(width):
        pixel = img.getpixel((x, y))
        if is_different(pixel, bg_color):
            different_pixels_in_row += 1
            if different_pixels_in_row >= 3:
                break
    if different_pixels_in_row >= 3:
        if first_y is None:
            first_y = y
        last_y = y

print(f"[*] Found content range: Y = {first_y} to {last_y}")

if first_y is not None and last_y is not None:
    # Calculate the actual content height and center
    content_height = last_y - first_y
    content_center = first_y + (content_height / 2.0)
    
    # Calculate symmetrical boundaries around the content center
    # Add a generous 40-pixel padding to both sides
    half_height = (content_height / 2.0) + 40
    
    crop_top = max(0, int(content_center - half_height))
    crop_bottom = min(height, int(content_center + half_height))
    
    # To enforce absolute symmetry, make the padding on both sides equal
    padding_top = first_y - crop_top
    padding_bottom = crop_bottom - last_y
    actual_padding = min(padding_top, padding_bottom)
    
    crop_top = first_y - actual_padding
    crop_bottom = last_y + actual_padding
    
    cropped_img = img.crop((0, crop_top, width, crop_bottom))
    cropped_img.save(repo_logo_path)
    
    print(f"[SUCCESS] Symmetrically cropped. Y-range: {crop_top} -> {crop_bottom} (New Height: {cropped_img.size[1]})")
    print(f"[*] Symmetrical padding on both sides: {actual_padding} pixels.")
else:
    print("[ERROR] Could not detect boundaries.")
