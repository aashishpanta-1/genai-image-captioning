import gradio as gr
from src.model import get_model_and_processor
from src.engine import generate_caption

# Load the model and processor once at the top of the file
model, processor, device = get_model_and_processor()

def predict(image):
    return generate_caption(image, model, processor, device)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs="text",
    title="Gen AI-Powered Image Captioning"
)

if __name__ == "__main__":
    interface.launch()