import hashlib
import hmac
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

import pytest
from hexus.webhook.dispatcher import sign_payload, dispatch_webhook_sync

class MockWebhookHandler(BaseHTTPRequestHandler):
    requests_received: List[Dict[str, Any]] = []
    response_status = 200

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        # Parse payload
        try:
            payload = json.loads(body.decode('utf-8'))
        except Exception:
            payload = None
            
        MockWebhookHandler.requests_received.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
            "payload": payload
        })
        
        self.send_response(MockWebhookHandler.response_status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        # Suppress logging to console during tests
        pass

@pytest.fixture
def mock_webhook_server():
    MockWebhookHandler.requests_received = []
    MockWebhookHandler.response_status = 200
    
    server = HTTPServer(('127.0.0.1', 0), MockWebhookHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/webhook"
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    
    yield url, MockWebhookHandler
    
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)

def test_sign_payload():
    payload = b'{"event":"test"}'
    secret = "my_secret"
    sig = sign_payload(payload, secret)
    
    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()
    assert sig == expected

@patch('time.sleep', return_value=None)
def test_dispatch_webhook_success(mock_sleep, mock_webhook_server):
    url, handler = mock_webhook_server
    secret = "secret_key"
    payload = {"hello": "world"}
    
    dispatch_webhook_sync(
        url=url,
        secret=secret,
        event="test_event",
        payload=payload,
        max_retries=2,
        initial_backoff=0.1
    )
    
    assert len(handler.requests_received) == 1
    req = handler.requests_received[0]
    
    assert req["headers"]["X-Hexus-Event"] == "test_event"
    assert "X-Hexus-Signature" in req["headers"]
    
    # Verify signature
    sig = req["headers"]["X-Hexus-Signature"]
    expected_sig = sign_payload(req["body"], secret)
    assert sig == expected_sig
    
    # Verify payload structure
    data = req["payload"]
    assert data["event"] == "test_event"
    assert data["data"] == payload
    assert "timestamp" in data

@patch('time.sleep', return_value=None)
def test_dispatch_webhook_retries_and_fails(mock_sleep, mock_webhook_server):
    url, handler = mock_webhook_server
    handler.response_status = 500
    
    dispatch_webhook_sync(
        url=url,
        secret=None,
        event="retry_event",
        payload={"some": "data"},
        max_retries=3,
        initial_backoff=0.01
    )
    
    # 1 initial attempt + 3 retries = 4 attempts total
    assert len(handler.requests_received) == 4
    for req in handler.requests_received:
        assert req["headers"]["X-Hexus-Event"] == "retry_event"
        assert "X-Hexus-Signature" not in req["headers"]

@patch('time.sleep', return_value=None)
@patch('hexus.MemoryProvider', new=object)
def test_provider_worker_dispatches_webhooks(mock_sleep, mock_webhook_server):
    url, handler = mock_webhook_server
    
    from hexus import HexusMemoryProvider
    from hexus.writer import _PendingWrite
    from hexus.webhook.dispatcher import dispatch_webhook_sync
    
    provider = HexusMemoryProvider(config={
        "webhook_url": url,
        "webhook_secret": "my_secret_key",
        "embed_on_write": False
    })
    provider._store = MagicMock()
    provider._session_id = "test-session"
    provider._agent_identity = "test-agent"
    
    # Mock _maybe_embed to return None to avoid any model dependency
    provider._maybe_embed = MagicMock(return_value=None)
    
    # Patch dispatch_webhook to run synchronously to avoid race conditions
    with patch('hexus.webhook.dispatcher.dispatch_webhook', side_effect=dispatch_webhook_sync):
        # 1. Test memory_retain from add action
        item_add = _PendingWrite(
            action="add",
            agent_identity="test-agent",
            target="user",
            content="Hello world memory text",
            metadata={"source": "test"}
        )
        provider._worker(item_add)
        
        assert len(handler.requests_received) == 1
        req = handler.requests_received[-1]
        assert req["headers"]["X-Hexus-Event"] == "memory_retain"
        assert req["payload"]["data"]["content"] == "Hello world memory text"
        
        # 2. Test memory_forget from remove action
        item_remove = _PendingWrite(
            action="remove",
            agent_identity="test-agent",
            target="user",
            content="Hello world memory text"
        )
        provider._worker(item_remove)
        
        assert len(handler.requests_received) == 2
        req = handler.requests_received[-1]
        assert req["headers"]["X-Hexus-Event"] == "memory_forget"
        assert req["payload"]["data"]["content"] == "Hello world memory text"
        
        # 3. Test memory_retain from replace action
        item_replace = _PendingWrite(
            action="replace",
            agent_identity="test-agent",
            target="user",
            content="Replaced content",
            metadata={"source": "test"}
        )
        provider._worker(item_replace)
        
        assert len(handler.requests_received) == 3
        req = handler.requests_received[-1]
        assert req["headers"]["X-Hexus-Event"] == "memory_retain"
        assert req["payload"]["data"]["content"] == "Replaced content"


@patch('time.sleep', return_value=None)
@patch.dict('os.environ', {})
def test_mcp_tools_dispatch_webhooks(mock_sleep, mock_webhook_server):
    url, handler = mock_webhook_server
    
    from hexus.store import MemoryStore
    from mcp_server import tools
    import os
    
    # Setup test environment DSN
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
        
    store = MemoryStore(dsn)
    store.ensure_schema()
    agent = "pytest-mcp-webhook-" + os.urandom(4).hex()
    
    # Configure webhooks via environment variables
    with patch.dict('os.environ', {
        "HEXUS_WEBHOOK_URL": url,
        "HEXUS_WEBHOOK_SECRET": "mcp_secret",
        "HEXUS_AGENT_IDENTITY": agent
    }):
        with patch('hexus.webhook.dispatcher.dispatch_webhook', side_effect=dispatch_webhook_sync):
            # 1. Test memory_retain tool triggers webhook
            with patch('mcp_server.tools._embed_batch', return_value=[[0.1] * 384]):
                res = tools.memory_retain(store, {
                    "contents": ["Hello from MCP server webhook test"],
                    "target": "memory",
                    "agent_identity": agent
                })
                assert res["inserted"] == 1
                
                assert len(handler.requests_received) == 1
                req = handler.requests_received[-1]
                assert req["headers"]["X-Hexus-Event"] == "memory_retain"
                assert req["payload"]["data"]["content"] == "Hello from MCP server webhook test"
                
                # Find the row ID to delete it
                with store._get_pool().connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id FROM memory_entries WHERE agent_identity = %s", (agent,))
                        row_id = cur.fetchone()[0]
                
                # 2. Test memory_forget tool triggers webhook
                forget_res = tools.memory_forget(store, {
                    "id": row_id,
                    "confirm": True,
                    "agent_identity": agent
                })
                assert forget_res["deleted"] == 1
                
                assert len(handler.requests_received) == 2
                req = handler.requests_received[-1]
                assert req["headers"]["X-Hexus-Event"] == "memory_forget"
                assert req["payload"]["data"]["content"] == "Hello from MCP server webhook test"

