# BurnScope

An AI-assisted burn assessment research prototype built with Streamlit. BurnScope analyzes a burn photo, estimates severity, highlights the wound, and provides source-cited care guidance through a conversational agent.

> Research prototype only. Not a medical device or a diagnosis; it has not been clinically validated and does not replace professional medical care.

## Features

- Photo quality checks and confidence-aware severity classification.
- Wound segmentation and Grad-CAM visual explanations.
- Source-cited guidance using retrieval-augmented generation (RAG).
- Nearby clinic search, follow-up reminders, and PDF reports with explicit consent.
- User accounts with case history and data deletion.

## Tech stack

Python · Streamlit · Groq · EfficientNetV2-S · SegFormer-B1 · TensorFlow · PyTorch · ChromaDB · SQLite / PostgreSQL

## Model results

| Evaluation | Result |
|---|---|
| Severity classification — in-domain test | 91.42% accuracy |
| Severity classification — independent external dataset | 75.25% accuracy |
| Wound segmentation — held-out test | 0.8479 Dice / 0.7530 IoU |

Performance drops on unfamiliar images. Confidence filtering can still allow incorrect predictions, and wound area is relative rather than a physical measurement. This is intended for research and demonstration use; account security is demonstration-grade and stored data is not encrypted at rest.

## Quick start

Requires Python 3.13, Git LFS, and a Groq API key.

```bash
git lfs install
git clone https://github.com/Aayush-patwa/Burnscope.git
cd Burnscope
python -m venv .venv
```

Activate the environment:

```powershell
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install dependencies and launch:

```bash
pip install -r requirements.txt
```

Copy [`.env.example`](.env.example) to `.env` and set `GROQ_API_KEY`, then run:

```bash
python -m streamlit run app.py
```

Model weights are included through Git LFS. Guidance documents and demo photos are not included; see the [guidance setup](rag/source_docs/README.md) and [sample image notes](data/sample_images/README.md). Without guidance documents, care guidance is unavailable. Optional database, clinic filtering, and email settings are documented in [`.env.example`](.env.example).

Run the automated tests with `python -m pytest`. See an [example PDF report](docs/sample_report.pdf).

## Data and license

Models use the [Skin Burn Dataset](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset), [Burn Wound Classification](https://universe.roboflow.com/binussss/burn-wound-classification/dataset/2), and [Burn Detection 1](https://universe.roboflow.com/nabil-cvnta/burn-detection-1) datasets.

Code is licensed under [MIT](LICENSE). Datasets, model weights, and guidance sources are subject to their respective terms.
