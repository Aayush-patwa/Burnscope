# BurnScope

An AI-assisted burn assessment research prototype built with Streamlit. BurnScope analyzes a burn photo, predicts a severity class, estimates relative wound extent, and provides source-cited care guidance through a conversational agent.

> **Research prototype only.** BurnScope is not a medical device or a diagnosis. It has not been clinically validated and does not replace professional medical care.

## Features

- Photo quality checks and confidence-aware severity classification with an abstention option.
- Wound segmentation for relative-area estimation and Grad-CAM visual explanations.
- Source-cited guidance using retrieval-augmented generation (RAG).
- Nearby clinic search and follow-up scheduling with optional email reminders.
- PDF report previews and consent prompts for report generation and sharing.
- User accounts, case history, and account/data deletion.

## Tech stack

Python · Streamlit · Groq · EfficientNetV2-S · SegFormer-B1 · TensorFlow · PyTorch · ChromaDB · SQLite / PostgreSQL

## Model results

| Evaluation | Test set | Result |
|---|---|---|
| Severity classification — in-domain | 734 images | 91.42% accuracy |
| Severity classification — independent external dataset | 408 images | 75.25% accuracy |
| Wound segmentation — held-out test | 204 images, split by source photo | 0.8479 Dice / 0.7530 IoU |

The external classification dataset was not used for training. At a confidence threshold of **0.90**, the classifier accepted **58.1%** of external images with **86.9% accuracy** on that accepted subset. However, **31 of 101 external-test errors still passed the gate**. Confidence filtering reduces some errors but does not guarantee a correct prediction.

These results evaluate the individual models; they do not establish clinical accuracy or validate the application's care recommendations.

## Limitations and privacy

- Performance drops on unfamiliar images, and performance across skin tones has not been validated.
- The quality check does not establish whether an image actually shows a burn.
- Wound area is a fraction of the photographed image, not a physical measurement or total body surface area. Changes in framing can affect follow-up comparisons.
- Segmentation currently estimates relative extent; a displayed wound outline is not yet part of the reviewed implementation.
- Analysis-state handling, workflow checks, and report-consent controls need further hardening before use beyond a controlled demonstration.
- Account security is demonstration-grade, and stored data is not encrypted at rest. Persistent database records do not make locally stored photos and reports durable on temporary hosting disks.
- Reminder delivery depends on the application running and email being configured.

Uploaded photos are re-encoded to remove embedded metadata and processed by local models. Groq receives conversational text and structured tool results rather than the uploaded photo. Optional clinic-search and email integrations communicate with external services.

## Quick start

Developed and tested with **Python 3.13**. Requires **Git LFS** and a **Groq API key**.

```bash
git lfs install
git clone https://github.com/Aayush-patwa/Burnscope.git
cd Burnscope
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy [`.env.example`](.env.example) to `.env`, set `GROQ_API_KEY`, and launch:

```bash
python -m streamlit run app.py
```

Model weights are included through Git LFS. **Guidance documents and demo photos are not included.** See the [guidance setup](rag/source_docs/README.md) and [sample image notes](data/sample_images/README.md). Without guidance documents, source-cited care guidance is unavailable.

Optional database, clinic filtering, and email settings are documented in [`.env.example`](.env.example). See an [example PDF report](docs/sample_report.pdf).

## Tests

```bash
python -m pytest
```

Tests requiring unavailable guidance documents or sample photos may be skipped. The separate `python tests/live_e2e.py` script makes real Groq calls and consumes API quota.

## Data and license

The models were trained and evaluated using:

- [Skin Burn Dataset](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset) — classification training and in-domain testing.
- [Burn Wound Classification](https://universe.roboflow.com/binussss/burn-wound-classification/dataset/2) — external classification evaluation.
- [Burn Detection 1](https://universe.roboflow.com/nabil-cvnta/burn-detection-1) — segmentation.

Code is licensed under [MIT](LICENSE). Datasets, model weights, and guidance sources are subject to their respective terms; check those terms before reuse or redistribution.
