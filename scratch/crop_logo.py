import os
from PIL import Image

logo_path = r"c:\Users\Gebruiker\Documents\Computational Bio\Somasays\somasays_pixel_logo.png"

if not os.path.exists(logo_path):
    print(f"[ERROR] Logo not found at {logo_path}")
    exit(1)

img = Image.open(logo_path)
# Ensure image is in RGB/RGBA
if img.mode not in ("RGB", "RGBA"):
    img = img.convert("RGB")

width, height = img.size

# Get background color from top-left pixel
bg_color = img.getpixel((0, 0))
print(f"[*] Image size: {width}x{height}, detected background color: {bg_color}")

# Scan rows to find content boundaries
first_y = None
last_y = None

# We check if a pixel is significantly different from the background color
# to handle any compression artifacts or slight gradients, though it should be solid.
def is_different(c1, c2, tolerance=15):
    return sum(abs(a - b) for a, b in zip(c1[:3], c2[:3])) > tolerance

for y in range(height):
    row_has_content = False
    for x in range(width):
        pixel = img.getpixel((x, y))
        if is_different(pixel, bg_color):
            row_has_content = True
            break
    if row_has_content:
        if first_y is None:
            first_y = y
        last_y = y

if first_y is not None and last_y is not None:
    # Add a clean minimal padding of 35 pixels
    padding = 35
    crop_top = max(0, first_y - padding)
    crop_bottom = min(height, last_y + padding)
    
    # Let's crop the image vertically, keeping full width
    cropped_img = img.crop((0, crop_top, width, crop_bottom))
    cropped_img.save(logo_path)
    print(f"[SUCCESS] Cropped vertically. Height reduced from {height} to {cropped_img.size[1]} (Y: {crop_top} -> {crop_bottom})")
else:
    print("[ERROR] Could not detect logo content boundaries.")
