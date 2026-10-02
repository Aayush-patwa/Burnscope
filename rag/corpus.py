"""
Optional download of the guidance documents from a PRIVATE GitHub repository.

The documents in rag/source_docs/ are adapted from third-party sources and are not part of the public
repository. A deployment (for example the live demo) can still use them: keep them in a private repo, give the
app a read-only token, and it fetches them once at start-up.

    GUIDANCE_REPO   owner/name of the private repo that holds the four .docx files, e.g. "someone/burnscope-guidance"
    GUIDANCE_TOKEN  a fine-grained personal access token with read-only "Contents" access to that one repo

Both must be set; otherwise nothing is downloaded. The token is never printed or logged.
"""

import io
import os
import zipfile

import requests

SOURCE_DIR = os.path.join(os.path.dirname(__file__), "source_docs")


def _has_documents() -> bool:
    return os.path.isdir(SOURCE_DIR) and any(f.lower().endswith(".docx") for f in os.listdir(SOURCE_DIR))


def fetch_corpus_if_configured() -> bool:
    """Download the .docx files into rag/source_docs/ when a private repo is configured and no documents are
    present yet. Returns True if documents were downloaded."""
    repo = os.environ.get("GUIDANCE_REPO", "").strip()
    token = os.environ.get("GUIDANCE_TOKEN", "").strip()
    if not repo or not token or _has_documents():
        return False
    try:
        response = requests.get(
            f"https://api.github.com/repos/{repo}/zipball",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
            timeout=60,
        )
        if response.status_code != 200:
            print(f"  [corpus] Could not download the guidance documents (HTTP {response.status_code}). "
                  "Check GUIDANCE_REPO and that the token has read access to it.")
            return False
        os.makedirs(SOURCE_DIR, exist_ok=True)
        saved = 0
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".docx") and not name.endswith("/"):
                    with open(os.path.join(SOURCE_DIR, os.path.basename(name)), "wb") as out:
                        out.write(archive.read(name))
                    saved += 1
        print(f"  [corpus] Downloaded {saved} guidance document(s) from the private repository.")
        return saved > 0
    except (requests.RequestException, zipfile.BadZipFile, OSError) as e:
        print(f"  [corpus] Could not download the guidance documents: {type(e).__name__}")
        return False
