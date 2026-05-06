import torch

def generate_caption_from_image(image, model, processor, device):
    """Generate a caption for a PIL image using the provided model and processor.

    Args:
        image (PIL.Image.Image): Input image.
        model (torch.nn.Module): Loaded captioning model.
        processor (transformers.PreTrainedProcessor): Processor for preparing the image.
        device (str): "cuda" or "cpu" where the model resides.
    Returns:
        str: Generated caption.
    """
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Compatibility wrapper expected by app.py
def generate_caption(image, model, processor, device):
    """Alias for generate_caption_from_image to maintain backward compatibility."""
    return generate_caption_from_image(image, model, processor, device)
