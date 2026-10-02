"""Run every test from the project root so relative paths (data/, models/, rag/) resolve."""
import os
import sys

import pytest

# Tests must never touch the real Postgres; load_dotenv() won't override a variable that is already set.
os.environ["DATABASE_URL"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


# A handful of checks run the real image tools on the demo photos in data/sample_images. Those photos come from
# third-party sources and are not part of the public repository, so on a fresh checkout these checks are skipped
# (they pass whenever the folder holds JPG photos of your own, named burnWound*.jpg as before).
_NEEDS_DEMO_PHOTOS = {
    "test_consent_medication_size.py": ("E1 ",),
    "test_loop_and_store.py": ("C3a", "C3b", "C3c", "C3d"),
    "test_quality_segment_retrieval_flow.py": ("Q2 ", "G1 ", "F1 ", "F5 "),
}


# Likewise, the guidance documents in rag/source_docs are adapted from third-party sources and are not part of the
# public repository, so tests that check what retrieval returns are skipped when they are absent.
_NEEDS_GUIDANCE_DOCS = {
    "test_consent_medication_size.py": ("D1 ",),
    "test_loop_and_store.py": ("C5 ",),
    "test_quality_segment_retrieval_flow.py": ("R1 ",),
}
_NEEDS_GUIDANCE_DOCS_BY_NAME = {
    "test_rag_and_report.py": {
        "test_top_passage_is_the_right_section", "test_patient_path_is_patient_safe_and_drug_free",
        "test_index_has_expected_metadata", "test_report_pdf_is_built_with_all_sections",
        "test_clinician_reference_is_really_found_for_every_severity", "test_real_pdf_contains_exactly_the_chosen_sections",
    },
}


def _demo_photos_present():
    folder = os.path.join(ROOT, "data", "sample_images")
    return os.path.isdir(folder) and any(f.lower().endswith(".jpg") for f in os.listdir(folder))


def _guidance_docs_present():
    folder = os.path.join(ROOT, "rag", "source_docs")
    return os.path.isdir(folder) and any(f.lower().endswith(".docx") for f in os.listdir(folder))


def pytest_collection_modifyitems(config, items):
    skip_photos = pytest.mark.skip(reason="needs demo photos in data/sample_images (not included in the public repository)")
    skip_docs = pytest.mark.skip(reason="needs the guidance documents in rag/source_docs (not included in the public repository)")
    photos, docs = _demo_photos_present(), _guidance_docs_present()
    for item in items:
        fname = os.path.basename(str(item.fspath))
        if not photos:
            for prefix in _NEEDS_DEMO_PHOTOS.get(fname, ()):
                if f"test_check[{prefix}" in item.nodeid:
                    item.add_marker(skip_photos)
        if not docs:
            for prefix in _NEEDS_GUIDANCE_DOCS.get(fname, ()):
                if f"test_check[{prefix}" in item.nodeid:
                    item.add_marker(skip_docs)
            if getattr(item, "originalname", item.name) in _NEEDS_GUIDANCE_DOCS_BY_NAME.get(fname, ()):
                item.add_marker(skip_docs)


@pytest.fixture(scope="session", autouse=True)
def _guidance_index():
    """The generated search index is not kept in git, so a fresh checkout builds it once before the tests run."""
    from rag.build_index import ensure_index
    ensure_index()
