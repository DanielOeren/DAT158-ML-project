import json
import numpy as np
import tensorflow as tf
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fruit_model.h5"
CLASS_PATH = ROOT / "class_names.json"

# Load artifacts
if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    raise SystemExit("Missing model artifacts. Train first: python src/train.py")

model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = json.load(open(CLASS_PATH))
IMG_H, IMG_W = model.input_shape[1:3]

def predict(img):
    # img is an HxWxC array (uint8 RGB) from Gradio
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (IMG_H, IMG_W))
    x = tf.keras.applications.efficientnet.preprocess_input(tf.expand_dims(img, 0))
    probs = model.predict(x, verbose=0)[0]
    # Return top-5 for a nice label display
    top5_idx = np.argsort(probs)[-5:][::-1]
    return {CLASS_NAMES[i]: float(probs[i]) for i in top5_idx}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a fruit photo"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 predictions"),
    title="Fruits-360 Classifier",
    description="Upload a fruit image and get the predicted class (trained on Fruits-360 100×100)."
)

if __name__ == "__main__":
    demo.launch()
