# Guidance documents (not included)

The retrieval module answers treatment questions only from a curated set of guidance documents. The documents
used by the authors are adapted from WHO material, national treatment guidelines and patient-information
leaflets, each published under its own licence, so they are **not redistributed in this repository**.

Without them the app still starts and works, but treatment questions get "no guidance available".

## Supplying your own

Put four Word (`.docx`) files in this folder, using exactly these names (they are listed in
`rag/build_index.py`):

| File | Topic |
|---|---|
| `Dressing Selection & Application.docx` | dressing types and how to apply them |
| `Immediate First-Aid & Safety.docx` | first aid, cooling, chemical and electrical burns |
| `Rehabilitation__Scar___Deformity_Prevention.docx` | rehabilitation, scars, nutrition |
| `Triage, Severity Assessment.docx` | burn depth, triage, infection signs |

Then build the index (it is also built automatically the first time the app starts):

```bash
python -m rag.build_index
```

## Document conventions

`rag/chunker.py` reads the Word structure directly. Its module docstring is the full reference; in short:

- A paragraph starting with `Section:` begins a new section.
- Bold, numbered or "Label:" sub-headings start a new passage; each table becomes one passage.
- `[SOURCE: Document - Chapter - pages]` records where a passage comes from. It is shown with every answer.
- `[AUDIENCE: clinician_only]` (or `patient_safe`) controls who may see a passage. Any passage containing a
  numeric drug or fluid dose is forced to `clinician_only` by code, whatever the document says.
- A paragraph that is only `[CHUNK]` forces a passage boundary.

Use sources you are allowed to reuse, keep the citation on every passage, and check the licence of each source
before publishing the result.
