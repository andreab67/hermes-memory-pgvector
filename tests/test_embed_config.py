"""Configurable embedding contract (v0.5.3): embed_dim, embed_api_key_env,
embed_protocol.

No DB, no network: urllib.request.urlopen is replaced with a recorder that
answers per URL path, so these tests see the exact URL, headers and body the
plugin would have sent.

Why this exists. The reference deployment moved its vector columns from
vector(768) to vector(1536) and re-embedded with an OpenRouter-hosted
OpenAI model. The plugin could not follow through config alone: _post()
rejected every vector that was not exactly 768 long, backfill refused non-768
probes, and no Authorization header was ever sent. The defaults must still
reproduce the old behaviour exactly -- 768 dims, no auth, auto protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pgvector_pkg  # noqa: E402
from hermes_pgvector import DEFAULTS, PgvectorMemoryProvider  # noqa: E402

# `hermes_pgvector.embed` is the re-exported FUNCTION after a normal import;
# import_module reaches the submodule through sys.modules.
embed_mod = import_module("hermes_pgvector.embed")

OPENAI_PATH = "/v1/embeddings"
OLLAMA_PATH = "/api/embed"
KEY_ENV = "HPGV_TEST_EMBED_KEY"
SECRET = "sk-test-DO-NOT-LEAK-0123456789"


class _Resp:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHTTP:
    """Stands in for urllib.request.urlopen.

    `routes` maps a URL path suffix to either a vector length (answer with a
    vector of that length, in that protocol's response shape) or an exception
    instance to raise. Unrouted URLs get a 404, like a server that does not
    speak that protocol.
    """

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        for suffix, action in self.routes.items():
            if req.full_url.endswith(suffix):
                if isinstance(action, BaseException):
                    raise action
                vec = [0.01] * action
                if suffix == OPENAI_PATH:
                    return _Resp({"data": [{"embedding": vec}]})
                return _Resp({"embeddings": [vec]})
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    @property
    def urls(self):
        return [r.full_url for r in self.requests]

    def auth_headers(self):
        return [r.get_header("Authorization") for r in self.requests]


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Any request not routed through _install() fails the test loudly."""

    def _refuse(req, timeout=None):
        raise AssertionError(f"unexpected real HTTP request to {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)


def _install(monkeypatch, routes) -> _FakeHTTP:
    fake = _FakeHTTP(routes)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _http_error(code: int, msg: str):
    return urllib.error.HTTPError("http://x", code, msg, None, None)


# --- defaults are the pre-0.5.3 behaviour ----------------------------------


def test_defaults_are_backward_compatible():
    assert DEFAULTS["embed_dim"] == 768
    assert not DEFAULTS["embed_api_key_env"]
    assert DEFAULTS["embed_protocol"] == "auto"
    assert embed_mod.DEFAULT_EMBED_DIM == 768


def test_default_dim_still_accepts_768(monkeypatch):
    _install(monkeypatch, {OPENAI_PATH: 768})
    vec = embed_mod.embed("hello", base_url="http://e")
    assert len(vec) == 768


def test_default_dim_still_rejects_other_lengths(monkeypatch):
    _install(monkeypatch, {OPENAI_PATH: 1536})
    with pytest.raises(embed_mod.EmbeddingDimensionError) as ei:
        embed_mod.embed("hello", base_url="http://e")
    assert "expected 768" in str(ei.value)
    assert "1536" in str(ei.value)


# --- embed_dim --------------------------------------------------------------


def test_dim_1536_accepts_1536(monkeypatch):
    _install(monkeypatch, {OPENAI_PATH: 1536})
    assert len(embed_mod.embed("hello", base_url="http://e", dim=1536)) == 1536


def test_dim_1536_rejects_768(monkeypatch):
    _install(monkeypatch, {OPENAI_PATH: 768})
    with pytest.raises(embed_mod.EmbeddingDimensionError) as ei:
        embed_mod.embed("hello", base_url="http://e", dim=1536)
    assert "expected 1536" in str(ei.value), "the message must state the expected value"


def test_dim_mismatch_still_does_not_fall_back_to_the_other_protocol(monkeypatch):
    """A wrong-dimension answer means the endpoint works and the model is
    wrong; auto mode must surface that, not a 404 from /api/embed."""
    fake = _install(monkeypatch, {OPENAI_PATH: 768, OLLAMA_PATH: 1536})
    with pytest.raises(embed_mod.EmbeddingDimensionError):
        embed_mod.embed("hello", base_url="http://e", dim=1536)
    assert not any(u.endswith(OLLAMA_PATH) for u in fake.urls)


@pytest.mark.parametrize("bad", [0, -1, "many", None])
def test_invalid_dim_is_an_embedding_error_not_a_crash(monkeypatch, bad):
    _install(monkeypatch, {OPENAI_PATH: 768})
    with pytest.raises(embed_mod.EmbeddingError):
        embed_mod.embed("hello", base_url="http://e", dim=bad)


# --- embed_api_key_env ------------------------------------------------------


def test_no_authorization_header_by_default(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)  # present in env, but not named
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    embed_mod.embed("hello", base_url="http://e")
    assert fake.auth_headers() == [None]


def test_authorization_header_uses_the_named_env_var(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    assert fake.auth_headers() == [f"Bearer {SECRET}"]


def test_api_key_is_read_at_call_time_not_cached(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    monkeypatch.setenv(KEY_ENV, "first-key")
    embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    monkeypatch.setenv(KEY_ENV, "rotated-key")
    embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    assert fake.auth_headers() == ["Bearer first-key", "Bearer rotated-key"]


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_empty_env_var_sends_no_header(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(KEY_ENV, value)
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    assert fake.auth_headers() == [None]


def test_header_is_sent_on_the_ollama_path_too(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OLLAMA_PATH: 768})
    embed_mod.embed(
        "hello", base_url="http://e", api_key_env=KEY_ENV, protocol="ollama"
    )
    assert fake.auth_headers() == [f"Bearer {SECRET}"]


def test_key_never_appears_in_an_auth_failure(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    _install(monkeypatch, {OPENAI_PATH: _http_error(401, "Unauthorized")})
    with pytest.raises(embed_mod.EmbeddingError) as ei:
        embed_mod.embed(
            "hello", base_url="http://e", api_key_env=KEY_ENV, protocol="openai"
        )
    assert "401" in str(ei.value)
    assert SECRET not in str(ei.value)
    assert SECRET not in repr(ei.value)


def test_key_never_leaks_through_a_rejected_header(monkeypatch):
    """http.client raises ValueError('Invalid header value ...') quoting the
    VALUE. That exception must not become the message, the cause, or the
    implicit context of what the plugin raises."""
    monkeypatch.setenv(KEY_ENV, SECRET)

    def _reject(req, timeout=None):
        raise ValueError(f"Invalid header value {req.get_header('Authorization')!r}")

    monkeypatch.setattr(urllib.request, "urlopen", _reject)
    with pytest.raises(embed_mod.EmbeddingError) as ei:
        embed_mod.embed(
            "hello", base_url="http://e", api_key_env=KEY_ENV, protocol="openai"
        )
    err = ei.value
    assert SECRET not in str(err)
    assert err.__cause__ is None
    assert err.__context__ is None


def test_key_with_a_control_character_is_refused_without_echoing_it(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-abc\ninjected")
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    with pytest.raises(embed_mod.EmbeddingError) as ei:
        embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    assert "sk-abc" not in str(ei.value)
    assert KEY_ENV in str(ei.value), "the variable NAME is what the operator needs"
    assert fake.requests == [], "nothing may be sent with a malformed key"


def test_key_is_not_logged(monkeypatch, caplog):
    monkeypatch.setenv(KEY_ENV, SECRET)
    _install(
        monkeypatch, {OPENAI_PATH: _http_error(401, "Unauthorized"), OLLAMA_PATH: 768}
    )
    with caplog.at_level(logging.DEBUG):
        embed_mod.embed("hello", base_url="http://e", api_key_env=KEY_ENV)
    assert SECRET not in caplog.text


# --- embed_protocol ---------------------------------------------------------


def test_protocol_openai_never_touches_api_embed(monkeypatch):
    fake = _install(
        monkeypatch, {OPENAI_PATH: _http_error(401, "Unauthorized"), OLLAMA_PATH: 768}
    )
    with pytest.raises(embed_mod.EmbeddingError) as ei:
        embed_mod.embed("hello", base_url="http://e", protocol="openai")
    assert "HTTP 401" in str(ei.value), (
        "the real error must surface, not a fallback 404"
    )
    assert fake.urls == ["http://e/v1/embeddings"]


def test_protocol_ollama_never_touches_v1_embeddings(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768, OLLAMA_PATH: 768})
    vec = embed_mod.embed("hello", base_url="http://e", protocol="ollama")
    assert len(vec) == 768
    assert fake.urls == ["http://e/api/embed"]


def test_protocol_auto_keeps_the_fallback(monkeypatch):
    fake = _install(monkeypatch, {OLLAMA_PATH: 768})  # /v1/embeddings -> 404
    vec = embed_mod.embed("hello", base_url="http://e", protocol="auto")
    assert len(vec) == 768
    assert fake.urls == ["http://e/v1/embeddings", "http://e/api/embed"]


def test_protocol_auto_prefers_openai_when_it_works(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768, OLLAMA_PATH: 768})
    embed_mod.embed("hello", base_url="http://e")
    assert fake.urls == ["http://e/v1/embeddings"]


def test_protocol_is_case_insensitive(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768, OLLAMA_PATH: 768})
    embed_mod.embed("hello", base_url="http://e", protocol=" Ollama ")
    assert fake.urls == ["http://e/api/embed"]


def test_unknown_protocol_falls_back_to_auto_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(embed_mod, "_warned_protocols", set())
    fake = _install(monkeypatch, {OLLAMA_PATH: 768})
    with caplog.at_level(logging.WARNING, logger=embed_mod.logger.name):
        embed_mod.embed("hello", base_url="http://e", protocol="grpc")
        embed_mod.embed("hello", base_url="http://e", protocol="grpc")
    assert fake.urls == ["http://e/v1/embeddings", "http://e/api/embed"] * 2
    warnings = [r for r in caplog.records if "grpc" in r.getMessage()]
    assert len(warnings) == 1, "warn once per bad value, not once per embed"


def test_read_timeout_is_an_embedding_error(monkeypatch):
    """A server that accepts the connection but answers slower than `timeout`
    raises a bare TimeoutError from getresponse() -- urllib does not wrap it.
    It must reach callers as EmbeddingError (their only except clause), and
    auto mode must still try the other protocol."""
    fake = _install(
        monkeypatch, {OPENAI_PATH: TimeoutError("timed out"), OLLAMA_PATH: 768}
    )
    assert len(embed_mod.embed("hello", base_url="http://e")) == 768
    assert len(fake.urls) == 2

    _install(monkeypatch, {OPENAI_PATH: TimeoutError("timed out")})
    with pytest.raises(embed_mod.EmbeddingError):
        embed_mod.embed("hello", base_url="http://e", protocol="openai")


# --- the provider routes config through one helper -------------------------

OPENROUTER_CFG = {
    "embed_url": "https://openrouter.ai/api",
    "embed_model": "openai/text-embedding-3-small",
    "embed_dim": 1536,
    "embed_api_key_env": KEY_ENV,
    "embed_protocol": "openai",
}


class _Store:
    def search(self, **kw):
        return [{"score": 0.9, "target": "memory", "content": "remembered fact"}]

    def hybrid_search(self, **kw):
        return self.search()

    def search_turns(self, **kw):
        return []

    def hybrid_search_turns(self, **kw):
        return []


def _provider(config) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider(config=config)
    p._healthy = True
    p._store = _Store()
    p._agent_identity = "marketing"
    p._raw_identity = "marketing"
    p._session_id = "sess-cfg"
    return p


def _assert_openrouter_request(req):
    assert req.full_url == "https://openrouter.ai/api/v1/embeddings"
    assert req.get_header("Authorization") == f"Bearer {SECRET}"
    body = json.loads(req.data)
    assert body["model"] == "openai/text-embedding-3-small"


def test_provider_prefetch_uses_the_openrouter_config(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OPENAI_PATH: 1536, OLLAMA_PATH: 1536})
    p = _provider(OPENROUTER_CFG)
    out = p.prefetch("what do we know")
    assert "remembered fact" in out
    assert len(fake.requests) == 1
    _assert_openrouter_request(fake.requests[0])


def test_every_provider_embed_path_honours_the_config(monkeypatch):
    """prefetch, both recall tools, the bulk-import closure and the writer
    drain must all send the same URL, model, header and dimension check."""
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OPENAI_PATH: 1536})
    p = _provider(OPENROUTER_CFG)

    p.prefetch("q")
    assert "error" not in json.loads(
        p.handle_tool_call("recall_memory", {"query": "q"})
    )
    assert "error" not in json.loads(
        p.handle_tool_call("recall_conversation", {"query": "q"})
    )
    assert len(p._make_embed_fn()("an entry")) == 1536
    assert len(p._maybe_embed("durable content")) == 1536

    assert len(fake.requests) == 5
    for req in fake.requests:
        _assert_openrouter_request(req)


def test_provider_with_1536_config_rejects_a_768_model(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    _install(monkeypatch, {OPENAI_PATH: 768})
    p = _provider(OPENROUTER_CFG)
    # Writer path: fail-soft to text-only (None), never raise.
    assert p._maybe_embed("durable content") is None
    # Hot path: prefetch degrades to no recall.
    assert p.prefetch("q") == ""


def test_provider_reads_the_key_at_call_time(monkeypatch):
    """Provider constructed BEFORE the variable exists (service env loaded
    late, or key rotated): the header must still be picked up."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    p = _provider(OPENROUTER_CFG)
    fake = _install(monkeypatch, {OPENAI_PATH: 1536})
    monkeypatch.setenv(KEY_ENV, SECRET)
    p.prefetch("q")
    assert fake.auth_headers() == [f"Bearer {SECRET}"]


def test_provider_string_config_values_are_coerced(monkeypatch):
    """save_config() persists schema values as STRINGS (invariant #9)."""
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OPENAI_PATH: 1536})
    p = _provider(
        {**OPENROUTER_CFG, "embed_dim": "1536", "embed_api_key_env": f" {KEY_ENV} "}
    )
    assert len(p._maybe_embed("durable content")) == 1536
    assert fake.auth_headers() == [f"Bearer {SECRET}"]


def test_provider_saved_empty_key_env_means_no_header(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    p = _provider(
        {"embed_api_key_env": "", "embed_dim": "768", "embed_protocol": "auto"}
    )
    assert len(p._maybe_embed("durable content")) == 768
    assert fake.auth_headers() == [None]


def test_provider_defaults_send_the_legacy_request(monkeypatch):
    fake = _install(monkeypatch, {OPENAI_PATH: 768})
    p = _provider(None)
    p.prefetch("q")
    assert fake.urls == [DEFAULTS["embed_url"] + OPENAI_PATH]
    assert fake.auth_headers() == [None]
    assert json.loads(fake.requests[0].data)["model"] == "nomic-embed-text"


@pytest.mark.parametrize("bad", ["", "wide", 0, -5, None])
def test_garbage_embed_dim_falls_back_to_the_default(bad):
    assert pgvector_pkg._embed_dim({"embed_dim": bad}) == 768


def test_new_keys_are_in_the_config_schema():
    schema = {e["key"]: e for e in PgvectorMemoryProvider().get_config_schema()}
    assert schema["embed_dim"]["default"] == "768"
    assert schema["embed_api_key_env"]["default"] == ""
    assert schema["embed_protocol"]["default"] == "auto"
    assert set(schema["embed_protocol"]["choices"]) == {"auto", "openai", "ollama"}
    assert "must return 768-dim" not in schema["embed_model"]["description"]


# --- CLI --------------------------------------------------------------------


def _cli():
    return import_module("hermes_pgvector.__main__")


def test_cli_backfill_embeds_with_the_config_file_settings(monkeypatch):
    cli = _cli()
    monkeypatch.setenv(KEY_ENV, SECRET)
    fake = _install(monkeypatch, {OPENAI_PATH: 1536})
    args = cli.build_parser().parse_args(["backfill"])
    fn = cli._make_embed_fn(args, dict(OPENROUTER_CFG))
    assert len(fn("a row being re-embedded")) == 1536
    _assert_openrouter_request(fake.requests[0])


def test_cli_flags_override_the_config_file(monkeypatch):
    cli = _cli()
    fake = _install(monkeypatch, {OPENAI_PATH: 1536, OLLAMA_PATH: 1536})
    args = cli.build_parser().parse_args(
        ["backfill", "--embed-dim", "1536", "--embed-protocol", "ollama"]
    )
    fn = cli._make_embed_fn(args, {"embed_dim": 768, "embed_protocol": "openai"})
    assert len(fn("text")) == 1536
    assert [u.endswith(OLLAMA_PATH) for u in fake.urls] == [True]


class _FakeStore:
    def __init__(self):
        self.backfill_kwargs = None

    def health(self):
        return {"ok": True}

    def count(self, **kw):
        return 0

    def count_turns(self, **kw):
        return 0

    def ensure_migration_002_applied(self):
        return False

    def backfill_null_embeddings(self, **kw):
        # Record only. Never call kw["embed_fn"] here: for `backfill` it is the
        # real embed closure, and the network must stay untouched.
        self.backfill_kwargs = kw
        return {"memory_entries": {"remaining": 0, "unembeddable": 0}}


def test_cli_stats_uses_the_configured_dim(monkeypatch, capsys):
    cli = _cli()
    store = _FakeStore()
    monkeypatch.setattr(cli, "_make_store", lambda args, cfg: store)
    monkeypatch.setattr(cli, "_load_config_file", lambda path: {"embed_dim": 1536})
    assert cli.cmd_stats(cli.build_parser().parse_args(["stats"])) == 0
    assert store.backfill_kwargs["expected_dim"] == 1536
    # The dry-run stand-in vector is the configured length, not a literal 768.
    assert len(store.backfill_kwargs["embed_fn"]("probe")) == 1536


def test_cli_backfill_passes_the_configured_dim_to_the_guard(monkeypatch, capsys):
    cli = _cli()
    store = _FakeStore()
    monkeypatch.setattr(cli, "_make_store", lambda args, cfg: store)
    monkeypatch.setattr(cli, "_load_config_file", lambda path: {"embed_dim": 1536})
    args = cli.build_parser().parse_args(["backfill", "--dry-run"])
    assert cli.cmd_backfill(args) == 0
    assert store.backfill_kwargs["expected_dim"] == 1536


def test_cli_backfill_defaults_to_768(monkeypatch, capsys):
    cli = _cli()
    store = _FakeStore()
    monkeypatch.setattr(cli, "_make_store", lambda args, cfg: store)
    monkeypatch.setattr(cli, "_load_config_file", lambda path: {})
    assert (
        cli.cmd_backfill(
            argparse.Namespace(
                config=None,
                dsn=None,
                embed_url=None,
                embed_model=None,
                tables=None,
                batch_size=100,
                dry_run=True,
            )
        )
        == 0
    )
    assert store.backfill_kwargs["expected_dim"] == 768


# --- store.backfill_null_embeddings expected_dim ---------------------------


class _PoolReached(Exception):
    """Raised by a stubbed pool: the dimension guard let the run through."""


def _offline_store(monkeypatch):
    MemoryStore = import_module("hermes_pgvector.store").MemoryStore
    s = MemoryStore("dbname=never-connected")

    def _no_db():
        raise _PoolReached()

    monkeypatch.setattr(s, "_get_pool", _no_db)
    return s


def test_backfill_default_expected_dim_is_768(monkeypatch):
    s = _offline_store(monkeypatch)
    with pytest.raises(ValueError) as ei:
        s.backfill_null_embeddings(
            embed_fn=lambda t: [0.1] * 1536, tables=("memory_entries",)
        )
    assert "expected 768" in str(ei.value)
    with pytest.raises(_PoolReached):
        s.backfill_null_embeddings(
            embed_fn=lambda t: [0.1] * 768, tables=("memory_entries",)
        )


def test_backfill_expected_dim_1536_accepts_1536_and_rejects_768(monkeypatch):
    s = _offline_store(monkeypatch)
    with pytest.raises(_PoolReached):
        s.backfill_null_embeddings(
            embed_fn=lambda t: [0.1] * 1536,
            tables=("memory_entries",),
            expected_dim=1536,
        )
    with pytest.raises(ValueError) as ei:
        s.backfill_null_embeddings(
            embed_fn=lambda t: [0.1] * 768,
            tables=("memory_entries",),
            expected_dim=1536,
        )
    assert "expected 1536" in str(ei.value)
    assert "768" in str(ei.value)
