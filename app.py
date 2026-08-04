import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

def find_artifact_directory():
    app_directory = Path(__file__).resolve().parent
    candidates = []
    configured_directory = os.environ.get("PV_FAULT_APP_DIR")
    if configured_directory:
        candidates.append(Path(configured_directory).expanduser())
    candidates.extend([
        app_directory,
        app_directory / "08_outputs" / "gradio_space",
        Path("/tmp/pv_fault_colab/outputs"),
        Path("/content/pv_fault_colab/outputs"),
    ])
    for candidate in candidates:
        if (candidate / "pv_fault_resnet18.pt").exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "The trained model was not found. Run the notebook first or set "
        "PV_FAULT_APP_DIR to the folder containing pv_fault_resnet18.pt.\n"
        f"Searched:\n{searched}"
    )


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


APP_DIR = find_artifact_directory()
CHECKPOINT = load_checkpoint(APP_DIR / "pv_fault_resnet18.pt")
LABEL_TO_INDEX = {
    str(label): int(index)
    for label, index in CHECKPOINT["label_to_index"].items()
}
INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}
LABEL_NAMES = [
    INDEX_TO_LABEL[index]
    for index in range(len(INDEX_TO_LABEL))
]
IMAGE_SIZE = int(CHECKPOINT.get("image_size", 128))
MEAN = CHECKPOINT.get("normalization_mean", [0.485, 0.456, 0.406])
STD = CHECKPOINT.get("normalization_std", [0.229, 0.224, 0.225])
CLASS_PRIORS = np.asarray(
    CHECKPOINT["class_priors"],
    dtype=np.float64,
).reshape(-1)
SELECTED_TAU = float(CHECKPOINT.get("selected_tau", 0.0))


MODEL = resnet18(weights=None)
MODEL.fc = nn.Linear(MODEL.fc.in_features, len(LABEL_NAMES))
MODEL.load_state_dict(CHECKPOINT["model_state_dict"])
MODEL.eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def _as_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


def _gradcam(input_tensor, target_index):
    activations = []
    gradients = []

    def capture_activations(module, inputs, output):
        activations.append(output)

    def capture_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = MODEL.layer4[-1]
    forward_handle = target_layer.register_forward_hook(capture_activations)
    backward_handle = target_layer.register_full_backward_hook(capture_gradients)
    try:
        with torch.enable_grad():
            MODEL.zero_grad(set_to_none=True)
            logits = MODEL(input_tensor.clone().detach())
            logits[0, target_index].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            cam = torch.relu(
                (weights * activations[0]).sum(dim=1)
            ).squeeze().detach().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()
        MODEL.zero_grad(set_to_none=True)

    cam -= cam.min()
    cam /= max(float(cam.max()), 1e-8)
    return cam


def _make_gradcam_figure(image, cam, label):
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    ).resize(image.size, Image.Resampling.BILINEAR)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[1].imshow(cam_image, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image)
    axes[2].imshow(cam_image, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Overlay: {label}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure


def classify_image(image):
    if image is None:
        raise gr.Error("Upload an infrared PV image.")

    image = _as_pil_image(image)
    input_tensor = TRANSFORM(image).unsqueeze(0)
    with torch.inference_mode():
        logits = MODEL(input_tensor)[0]
        baseline_probabilities = torch.softmax(logits, dim=0).numpy()
        adjusted_logits = logits - (
            SELECTED_TAU * torch.from_numpy(np.log(CLASS_PRIORS)).float()
        )
        adjusted_probabilities = torch.softmax(adjusted_logits, dim=0).numpy()

    baseline_index = int(np.argmax(baseline_probabilities))
    adjusted_index = int(np.argmax(adjusted_probabilities))
    baseline_label = INDEX_TO_LABEL[baseline_index]
    adjusted_label = INDEX_TO_LABEL[adjusted_index]

    probability_table = pd.DataFrame({
        "Class": LABEL_NAMES,
        "Baseline probability": baseline_probabilities,
        "Adjusted probability": adjusted_probabilities,
    }).sort_values(
        "Adjusted probability",
        ascending=False,
    ).reset_index(drop=True)
    probability_table[[
        "Baseline probability",
        "Adjusted probability",
    ]] = probability_table[[
        "Baseline probability",
        "Adjusted probability",
    ]].round(4)

    summary = (
        f"### Predictions\n"
        f"- Baseline: **{baseline_label}** "
        f"({baseline_probabilities[baseline_index]:.2%})\n"
        f"- Logit-adjusted: **{adjusted_label}** "
        f"({adjusted_probabilities[adjusted_index]:.2%})\n"
        f"- Selected tau: **{SELECTED_TAU:.1f}**"
    )
    cam = _gradcam(input_tensor, adjusted_index)
    figure = _make_gradcam_figure(image, cam, adjusted_label)
    plt.close(figure)
    return summary, probability_table, figure


with gr.Blocks(
    title="Fine-Grained Photovoltaic Fault Classification from Infrared Images"
) as demo:
    gr.Markdown(
        "# Fine-Grained Photovoltaic Fault Classification from Infrared Images\n"
        "Classify an infrared PV module image using the trained ResNet-18 model.\n\n"
        "*Research demonstrator only — not a substitute for professional inspection.*"
    )
    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload infrared PV image",
        )
        with gr.Column():
            predict_button = gr.Button(
                "Classify image",
                variant="primary",
            )
            summary_output = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Class probabilities"):
            probability_output = gr.Dataframe(
                label="Baseline and logit-adjusted probabilities",
                interactive=False,
            )
        with gr.Tab("Grad-CAM"):
            gradcam_output = gr.Plot(
                label="Model interpretation",
            )

    predict_button.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[
            summary_output,
            probability_output,
            gradcam_output,
        ],
        api_name="classify",
    )


if __name__ == "__main__":
    demo.launch()
