# Image Captioning with BLIP and PyTorch

## Overview
This repository contains a fully functional image‑captioning system that has been refactored from an exploratory Jupyter notebook into a modular Python application. The codebase separates model loading, inference logic, and the user interface, making it easier to maintain, test, and extend.

## Technical Specifications
- **Model**: Salesforce BLIP‑base (pre‑trained vision‑language transformer)
- **Framework**: PyTorch (automatic CUDA/GPU detection, fallback to CPU)
- **Interface**: Gradio web UI for uploading images and retrieving captions

## Project Structure
- `app.py` – entry point that launches the Gradio interface.
- `src/` – core Python package:
  - `model.py` – loads the BLIP processor and model, determines the device.
  - `engine.py` – contains the `generate_caption_from_image` function that runs inference.
  - `predict.py` – convenience wrapper for calling the engine from scripts or notebooks.
- `notebooks/` – original exploratory notebooks used during development.
- `requirements.txt` – pinned dependencies for reproducing the environment.

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/genai-image-captioning.git
cd genai-image-captioning

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell (use `venv\Scripts\activate.bat` on cmd)

# Install the required packages
pip install -r requirements.txt
```

## Running the Application
```bash
python app.py
```
The application will start a local Gradio server. Open a browser and navigate to `http://127.0.0.1:7860` to upload images and obtain captions.

## About the Author
**Aashish** – final‑year BSc. CSIT student at Patan Multiple Campus, Tribhuvan University.