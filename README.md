# BurnScope

BurnScope is a research prototype that helps a person decide what to do about a burn. From one smartphone
photo it checks the photo, estimates burn severity (with a built-in "I am not sure" option), outlines the wound,
shows what the model looked at, gives source-cited first-aid and care guidance, schedules a follow-up, and, only
with the user's explicit consent, prepares a PDF report for a clinician.

A tool-calling language-model agent (Groq, `openai/gpt-oss-120b`) coordinates eight tools. Code around the agent
enforces every rule that affects safety (consent, privacy, no medication advice), so those rules do not depend
on the model behaving.

> **BurnScope is not a medical device and does not diagnose.** It is a decision-support prototype that has not
> been clinically validated. It supports, and does not replace, evaluation by a qualified clinician. In an
> emergency, call your local emergency number.



---

## Results

| Measure | Result | Notes |
|---|---|---|
| Severity classifier, in-domain test (EfficientNetV2-S) | **91.42%** accuracy, 91.35% macro F1 | 734 test images from the same source as the training data |
| Severity classifier, external dataset | **75.25%** accuracy, 75.48% macro F1 | 408 images from an independent source, never used for training; the better guide to real-world performance |
| Reliability gate on the external set (confidence 0.90) | accepts 58.1% of images at **86.9%** accuracy; withholds 70 of 101 errors | 31 errors still pass the gate; calibration on new data is poor (ECE 0.127) |
| Wound segmentation (SegFormer-B1) | Dice **0.8479**, IoU **0.7530** | 204 held-out test images, split by source photo |
| Automated tests | 158 | scripted language model, real tools; no network needed |

The drop from 91% to 75% on unfamiliar images is a real limitation, not noise. See [Known limitations](#known-limitations).

---

## What it does

1. **Image quality check** (`tools/quality_check.py`): rule-based (no ML) test of size, brightness, contrast,
   sharpness and skin presence. A bad photo is rejected with a concrete instruction.
2. **Severity classification** (`tools/classify.py`): EfficientNetV2-S predicts 1st/2nd/3rd degree. The
   confidence is temperature-scaled (T = 1.1571) and a gate returns `PROCEED` or `ABSTAIN` at 0.90. An abstained
   prediction is never treated as established.
3. **Wound segmentation** (`tools/segment.py`): SegFormer-B1 returns the *relative* wound area (small, moderate or
   large) and whether there are several regions. It deliberately gives no square-centimetre value, because a
   photo has no scale reference.
4. **Grad-CAM** (`tools/gradcam.py`): a heatmap of the regions that influenced the prediction.
5. **Guidance retrieval** (`tools/rag_retrieve.py`, `rag/`): meaning-based search over a curated, source-cited
   guidance corpus. Chat only sees patient-safe passages, an unrelated question gets "no guidance found", and
   medication names and doses are removed from replies.
6. **Follow-up scheduling** (`tools/scheduler.py`): stores the next check-in date and e-mails a calendar invite.
7. **Nearby care** (`tools/clinic_locator.py`): OpenStreetMap search, filtered by a Gemini call that removes
   unsuitable facilities.
8. **Clinician report** (`tools/escalate.py`): a PDF that is previewed first and generated or e-mailed only after
   a clear, code-checked "yes". An example is in [`docs/sample_report.pdf`](docs/sample_report.pdf).

The agent reasons a triage verdict (self-care, clinical review or urgent care) from severity, calibrated
confidence and wound area. A deterministic function mirrors the same rules as a fallback.

**Accounts and history.** Each user's cases, photos and reports are private to their account. A signed-in user
stays signed in across page reloads (a revocable cookie, 14 days) and the open case reopens where they left it.
There is a History page, a Profile page (including "delete my data") and e-mail reminders for due follow-ups
while the app is running.

---

## Architecture

```
Streamlit UI (app.py)
        |
        v
Service layer (agent/service.py)   runs everything "as" the signed-in user (agent/userctx.py)
        |
        v
Agent loop (agent/loop.py)         calls Groq; each reply is a tool call or a final answer
        |
        |-- Tool schemas -> dispatch -> tools/*.py   (the eight tools)
        |-- Consent gate (agent/consent.py)          enforced in code, not only in the prompt
        |-- Medication filter (agent/medication_filter.py)
        |-- Context/size management (agent/context.py)
        `-- Case store (agent/case_store.py)         SQLite, or Postgres when DATABASE_URL is set

RAG: rag/source_docs/*.docx -> rag/chunker.py -> rag/vector_store.py (ChromaDB, all-mpnet-base-v2 embeddings)
```

**Design principles.** The agent decides what to do next; code guarantees its own rules (image-path lock,
forced segmentation before a 2nd/3rd-degree verdict, forced retrieval before care advice, consent gate, no false
"report sent" claims, an eight-tool-call cap). Nothing leaves the app without a preview and an explicit yes.
Every user's data is isolated by code. The language model never sees the photo, only typed text and structured
tool results.

---

## What is (and is not) in this repository

| Included | Not included |
|---|---|
| All source code (`app.py`, `agent/`, `tools/`, `rag/*.py`, `tests/`) | The **guidance documents** the retrieval module searches (adapted from third-party sources under their own licences). See [`rag/source_docs/README.md`](rag/source_docs/README.md) for what to supply |
| The two trained **model weights** (~162 MB, via Git LFS) | The **demo photos** (third-party images of real people). See [`data/sample_images/README.md`](data/sample_images/README.md) |
| An example report (`docs/sample_report.pdf`) | The generated search index (rebuilt automatically) |
| `.env.example` | Secrets (`.env`), the local database, per-user uploads and reports |

Without the guidance documents the app still starts and works, but treatment questions get "no guidance
available". Tests that need the photos or the documents are skipped automatically, so a fresh checkout passes.

---

## Setup

Requirements: **Python 3.13** (the version it was built and tested on), a free
[Groq API key](https://console.groq.com), [Git LFS](https://git-lfs.com) and about 500 MB of free disk space
(the model weights are ~162 MB; TensorFlow and PyTorch add more). It runs on Windows, macOS and Linux; it was
developed on Windows.

```bash
git lfs install                        # once per machine, before cloning, so the model weights download
git clone https://github.com/Aayush-patwa/Burnscope.git
cd Burnscope

python -m venv .venv
.venv\Scripts\activate                 # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                   # then put your GROQ_API_KEY in .env
python -m streamlit run app.py         # opens http://localhost:8501
```

The first sign-in on a fresh start takes about a minute while the models load. Create an account with any
name, e-mail and passcode (sign-in is demonstration-grade, see below).

There is also a plain terminal session: `python -m agent.loop`.

### Configuration (`.env`)

| Variable | Needed? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | The agent's language-model calls |
| `DATABASE_URL` | No | Postgres (e.g. a Neon pooled URL). When set, users, cases and sign-in sessions live there instead of local SQLite, so they survive restarts on hosts with a temporary disk |
| `GUIDANCE_REPO`, `GUIDANCE_TOKEN` | No | Fetch the guidance documents at start-up from a **private** GitHub repo (`owner/name` plus a read-only token), so a deployment can use them without publishing them |
| `GEMINI_API_KEY` | No | Clinic-facility filtering only; unfiltered results if unset |
| `EMAIL_ADDRESS`, `EMAIL_APP_PASSWORD` | No | Reminder and report e-mails (a Gmail App Password); silently disabled if unset |
| `HF_TOKEN` | No | Only if downloading the public SegFormer base model hits a rate limit |
| `BURNSCOPE_DB`, `BURNSCOPE_DEBUG`, `BURNSCOPE_HISTORY_TOKEN_BUDGET`, `BURNSCOPE_NO_WARMUP` | No | Developer overrides, documented in `.env.example` |

### Tests

```bash
pytest          # 158 tests, scripted language model, no Groq calls
```

`python tests/live_e2e.py` is a separate end-to-end script that uses real Groq calls (it costs quota) and is not
part of `pytest`.

---

## Project layout

```
agent/     agent loop, tool dispatch, consent gate, medication filter, accounts and isolation,
           case storage, reminders, and the service layer the UI calls
tools/     the eight tools: quality check, classifier, segmentation, Grad-CAM, retrieval,
           clinic locator, scheduler, escalation report
rag/       chunker, index builder and vector-store code (the documents themselves are not included)
models/    EfficientNetV2-S classifier (.keras) and SegFormer-B1 weights (.pth), stored with Git LFS
data/      runtime data (local database, per-user files), all git-ignored
docs/      an example generated report
tests/     pytest suite
app.py     the Streamlit web app
```

---

## Data and sources

The models were trained and evaluated on public datasets:

- Classification, training and in-domain test: [Skin Burn Dataset](https://www.kaggle.com/datasets/shubhambaid/skin-burn-dataset) (Kaggle).
- Classification, external validation (never used for training): [Burn Wound Classification](https://universe.roboflow.com/binussss/burn-wound-classification/dataset/2) (Roboflow Universe).
- Segmentation: [Burn Detection 1](https://universe.roboflow.com/nabil-cvnta/burn-detection-1) (Roboflow Universe).

Check each dataset's own licence before reusing the trained weights. The guidance corpus used by the authors is
adapted from the World Health Organization, the Government of India's burn treatment guidelines and the NPPMBI
handbook, plus public first-aid and paediatric information from ANZCOR, the Royal Children's Hospital Melbourne,
the NHS, Health NZ / KidsHealth NZ, Mass General Brigham and an ILCOR systematic review. Each passage carries
its citation in the app.

---

## Known limitations

- **Generalization.** Accuracy falls from 91.42% (familiar images) to 75.25% (an independent source). Do not
  assume it works on arbitrary photos.
- **The gate is a filter, not a guarantee.** About a third of errors on the external set passed it, mostly
  underestimating severity, and the confidence is overconfident on new data.
- **No clinical validation.** It has not been evaluated by clinicians or patients, and the accuracy of its triage
  verdicts has not been measured against clinician decisions.
- **The quality check cannot tell that a photo shows a burn**, and neither it nor the classifier has been
  validated across skin tones.
- **No real-world size.** Wound extent is relative; a photo has no scale reference.
- **The agent is not fully deterministic.** The order of tool calls and the wording vary between runs, which is
  why the safety checks are written as code.
- **Borderline photos can flip** between PROCEED and ABSTAIN from run to run because of CPU floating-point
  differences around the 0.90 threshold.
- **Sign-in is demonstration-grade**: no e-mail verification or password reset. Reminders only fire while the
  app is running.
- **Free-tier limits.** Groq's free tier is shared per organization; a full conversation costs roughly
  30,000-90,000 tokens, and photo turns can take from seconds to about two minutes.

## Privacy

Uploaded photos are re-encoded on receipt (which removes embedded metadata such as GPS position) and are only
ever processed by the local models; the language model sees typed text and structured results, never the image.
Each user's files are kept in their own folder, and "delete my data" removes the account, cases, photos and
reports. Nothing is encrypted at rest and there is no access control beyond the app's own account logic, so any
deployment beyond a local demo needs a proper security and data-protection review first.

## Licence

The code is released under the [MIT licence](LICENSE). The model weights, the datasets they were trained on, and
the guidance sources each have their own terms (see "Data and sources").
