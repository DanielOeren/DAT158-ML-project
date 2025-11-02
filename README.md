# DAT158-ML-project
Group project for DAT158

# Fruits-360 Classifier (VS Code)

End-to-end project:
- Download Fruits-360 (100x100 branch) via `kagglehub`
- Train a CNN (EfficientNetB0 transfer learning)
- Serve a Gradio web app for image → fruit prediction

## Quickstart
```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Download dataset into data/
python src/get_data.py

# 2) Train (produces fruit_model.keras & class_names.json at repo root)
python src/train.py

# 3) Run the app and upload an image
python src/app.py
