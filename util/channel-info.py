

from pathlib import Path

wsi = Path("G:\\placenta\\wsi\\20230815_095205_a.tiff")

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Load the image
image = Image.open(wsi)

# Get the number of channels
num_channels = len(image.getbands())
print(f'The image has {num_channels} channels.')