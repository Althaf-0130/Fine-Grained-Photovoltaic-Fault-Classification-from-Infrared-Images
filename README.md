# Fine-Grained Photovoltaic Fault Classification

Deep-learning classification of photovoltaic module faults from infrared images.

## Dataset

The project uses the [Raptor Maps InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) dataset:

- 20,000 infrared images
- 12 fault and operating-condition classes
- strong class imbalance

## Method

- Exact-duplicate filtering before data splitting
- Deterministic 70/15/15 training, validation, and test split
- ImageNet-pretrained ResNet-18
- Natural training distribution
- Validation-selected logit adjustment
- Held-out test evaluation
- Per-class metrics, confusion matrix, bootstrap intervals, and Grad-CAM

## Test Results

| Method | Accuracy | Balanced accuracy | Macro-F1 | Mean rare-class recall |
|---|---:|---:|---:|---:|
| Baseline | 0.7983 | 0.5700 | 0.6117 | 0.3999 |
| Logit adjustment (`tau = 0.6`) | 0.7957 | 0.6394 | 0.6338 | 0.5465 |

Logit adjustment improves macro-F1 by `0.0221`, balanced accuracy by `0.0693`, and mean rare-class recall by `0.1465`.

## Run in Google Colab

Open [PV_Fault_Classification_Colab.ipynb](PV_Fault_Classification_Colab.ipynb), select a T4 GPU, and choose **Runtime → Run all**.

The notebook downloads the dataset, trains and evaluates the model, displays all tables and figures, and downloads the generated files as a ZIP archive.

## Demo Application

After the notebook saves the model checkpoint, install the project requirements and run python app.py. The Gradio application accepts an infrared image and displays baseline and logit-adjusted predictions, class probabilities, and Grad-CAM. It is a research demonstrator and not a substitute for professional inspection.

## Generated Files

- Model checkpoint
- Experiment configuration
- Training history
- Split manifests
- Test predictions
- Overall and per-class metrics
- Bootstrap summary
- Confusion matrix
- Training and validation plots
- Grad-CAM visualization
