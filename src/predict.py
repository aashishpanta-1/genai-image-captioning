from PIL import Image
import torch
import sys, os
# Ensure the project root is in the Python path so that src modules can be imported when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import get_model_and_processor
from engine import generate_caption_from_image



# Load model and processor once at import time
model, processor, device = get_model_and_processor()


def generate_caption(image_path):
    """Generate a caption for an image file.

    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Generated caption.
    """
    image = Image.open(image_path).convert("RGB")
    caption = generate_caption_from_image(image, model, processor, device)
    return caption