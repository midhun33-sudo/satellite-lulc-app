import base64
import io
import os
from flask import Flask, jsonify, render_template, request
from model_utils import CLASSES, GradCAM, build_model, create_cam_overlay, get_transforms
from PIL import Image
import torch
import torch.nn.functional as F

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/eurosat_resnet18.pth"

# Initialize model
model = build_model(num_classes=len(CLASSES), pretrained=True)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded trained weights from {MODEL_PATH}")
else:
    print(
        "WARNING: Trained checkpoint not found. Using ImageNet initialization."
        " Run train.py first!"
    )

model.to(DEVICE)
model.eval()

# Hook into the final residual block of ResNet-18 for Grad-CAM
grad_cam = GradCAM(model, target_layer=model.layer4[-1])
transform = get_transforms()


def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Generate Grad-CAM and forward logits
        cam_mask, logits = grad_cam.generate(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        # Top-3 predictions
        top_indices = probs.argsort()[::-1][:3]
        predictions = [
            {"class": CLASSES[idx], "confidence": round(float(probs[idx]) * 100, 2)}
            for idx in top_indices
        ]

        # Build visual outputs
        resized_orig = pil_img.resize((224, 224))
        overlay_img = create_cam_overlay(pil_img, cam_mask)

        return jsonify({
            "top_prediction": predictions[0],
            "top_3": predictions,
            "original_image": pil_to_base64(resized_orig),
            "gradcam_image": pil_to_base64(overlay_img),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)