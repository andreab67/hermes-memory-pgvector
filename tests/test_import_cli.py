import json
import pytest
from unittest.mock import MagicMock
from hexus.store import MemoryStore
from mcp_server.import_cli import main

@pytest.fixture
def mock_store(monkeypatch):
    store = MagicMock(spec=MemoryStore)
    store.add.return_value = 1
    store.bulk_upsert_md.return_value = {"parsed": 5, "inserted": 3, "skipped": 2}
    
    # Mock embed to avoid torch loading or network calls
    monkeypatch.setattr("mcp_server.import_cli.embed", lambda x: [0.1] * 384)
    return store

def test_import_markdown(mock_store, tmp_path, monkeypatch):
    md_file = tmp_path / "test_memory.md"
    md_file.write_text("Entry 1\n§\nEntry 2\n")
    
    # Mock MemoryStore constructor to return our mock_store
    monkeypatch.setattr("mcp_server.import_cli.MemoryStore", lambda dsn: mock_store)
    
    # Run CLI main
    res = main(["--dsn", "dbname=test", "markdown", str(md_file)])
    assert res == 0
    mock_store.bulk_upsert_md.assert_called_once()
    mock_store.close.assert_called_once()

def test_import_mem0(mock_store, tmp_path, monkeypatch):
    json_file = tmp_path / "mem0.json"
    data = [
        {"memory": "Remember key details about client X", "metadata": {"category": "business"}},
        {"memory": "Likes black coffee"}
    ]
    json_file.write_text(json.dumps(data))
    
    monkeypatch.setattr("mcp_server.import_cli.MemoryStore", lambda dsn: mock_store)
    
    res = main(["--dsn", "dbname=test", "mem0", str(json_file)])
    assert res == 0
    assert mock_store.add.call_count == 2
    mock_store.close.assert_called_once()

def test_import_honcho(mock_store, tmp_path, monkeypatch):
    json_file = tmp_path / "honcho.json"
    data = [
        {"content": "Honcho exported statement 1", "id": "msg_001"},
        {"content": "Honcho exported statement 2", "id": "msg_002"}
    ]
    json_file.write_text(json.dumps(data))
    
    monkeypatch.setattr("mcp_server.import_cli.MemoryStore", lambda dsn: mock_store)
    
    res = main(["--dsn", "dbname=test", "honcho", str(json_file)])
    assert res == 0
    assert mock_store.add.call_count == 2
    mock_store.close.assert_called_once()

def test_import_holographic(mock_store, tmp_path, monkeypatch):
    json_file = tmp_path / "holo.json"
    data = [
        {"fact": "User is a backend engineer", "confidence": 0.9},
        {"fact": "Likes to code in Rust", "confidence": 0.8}
    ]
    json_file.write_text(json.dumps(data))
    
    monkeypatch.setattr("mcp_server.import_cli.MemoryStore", lambda dsn: mock_store)
    
    res = main(["--dsn", "dbname=test", "holographic", str(json_file)])
    assert res == 0
    assert mock_store.add.call_count == 2
    mock_store.close.assert_called_once()
