from tests.utils import generate_test_image
import sys
import os

# Get the parent directory and add it to sys.path
sys.path.append(os.path.abspath(".."))

print("\nTesting generating image\n")

img = generate_test_image(num_bands=120, dynamic_range=12)

print(f"Hyperspectral image shape: {img.shape}")


