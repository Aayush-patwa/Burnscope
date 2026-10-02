"""The guidance documents can be fetched from a private GitHub repo at start-up. No network: requests is faked."""
import io
import zipfile

import pytest

import rag.corpus as corpus


def _zip_with(*names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in names:
            z.writestr(name, b"fake docx bytes")
    return buf.getvalue()


class _Response:
    def __init__(self, status=200, content=b""):
        self.status_code, self.content = status, content


@pytest.fixture()
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "SOURCE_DIR", str(tmp_path / "source_docs"))
    return tmp_path / "source_docs"


def test_nothing_happens_when_not_configured(folder, monkeypatch):
    monkeypatch.delenv("GUIDANCE_REPO", raising=False)
    monkeypatch.delenv("GUIDANCE_TOKEN", raising=False)
    monkeypatch.setattr(corpus.requests, "get", lambda *a, **k: pytest.fail("must not call the network"))
    assert corpus.fetch_corpus_if_configured() is False


def test_downloads_only_docx_files_and_flattens_folders(folder, monkeypatch):
    monkeypatch.setenv("GUIDANCE_REPO", "someone/guidance")
    monkeypatch.setenv("GUIDANCE_TOKEN", "secret-token")
    seen = {}

    def fake_get(url, headers=None, timeout=None):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return _Response(200, _zip_with("someone-guidance-abc123/First Aid.docx", "someone-guidance-abc123/notes.txt",
                                        "someone-guidance-abc123/sub/Triage.docx"))
    monkeypatch.setattr(corpus.requests, "get", fake_get)
    assert corpus.fetch_corpus_if_configured() is True
    assert seen["url"] == "https://api.github.com/repos/someone/guidance/zipball" and seen["auth"] == "Bearer secret-token"
    assert sorted(p.name for p in folder.iterdir()) == ["First Aid.docx", "Triage.docx"]


def test_existing_documents_are_not_replaced(folder, monkeypatch):
    folder.mkdir()
    (folder / "mine.docx").write_bytes(b"x")
    monkeypatch.setenv("GUIDANCE_REPO", "someone/guidance")
    monkeypatch.setenv("GUIDANCE_TOKEN", "secret-token")
    monkeypatch.setattr(corpus.requests, "get", lambda *a, **k: pytest.fail("must not call the network"))
    assert corpus.fetch_corpus_if_configured() is False


def test_failures_are_reported_without_leaking_the_token(folder, monkeypatch, capsys):
    monkeypatch.setenv("GUIDANCE_REPO", "someone/guidance")
    monkeypatch.setenv("GUIDANCE_TOKEN", "secret-token")
    monkeypatch.setattr(corpus.requests, "get", lambda *a, **k: _Response(404))
    assert corpus.fetch_corpus_if_configured() is False
    out = capsys.readouterr().out
    assert "404" in out and "secret-token" not in out

    def boom(*a, **k):
        raise corpus.requests.ConnectionError("secret-token in message")
    monkeypatch.setattr(corpus.requests, "get", boom)
    assert corpus.fetch_corpus_if_configured() is False
    assert "secret-token" not in capsys.readouterr().out
