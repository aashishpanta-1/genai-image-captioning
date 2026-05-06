import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_model_and_processor():
    """Load the BLIP image captioning model and processor.

    Returns:
        model (torch.nn.Module): The model loaded onto the appropriate device.
        processor (BlipProcessor): Processor for preparing images.
        device (str): "cuda" if GPU is available, otherwise "cpu".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor, device
        