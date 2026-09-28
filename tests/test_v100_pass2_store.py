"""Pass-2 review fixes in store.py / embed.py (P2STORE-1, -2, -3A, -3B)."""

from __future__ import annotations

import http.server
import json
import os
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector.embed import EmbeddingDimensionError, EmbeddingError, embed  # noqa: E402
from hermes_pgvector.store import MemoryStore  # noqa: E402


@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-p2store-" + os.urandom(4).hex()
    yield s, agent
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()


# --- P2STORE-1: non-UTF-8 bytes on SQL_ASCII ---------------------------------

def test_backfill_undecodable_bytes_counted_failed_not_skipped_changed():
    import psycopg
    admin, runtime = os.environ.get("PG_TEST_ADMIN_DSN"), os.environ.get("PG_TEST_DSN")
    if not (admin and runtime):
        pytest.skip("PG_TEST_ADMIN_DSN / PG_TEST_DSN not set")

    def with_db(dsn, name):
        return re.sub(r"dbname=\S+", f"dbname={name}", dsn)

    name = "pytest_p2enc_" + os.urandom(3).hex()
    maint = with_db(admin, "postgres")
    with psycopg.connect(maint, autocommit=True) as conn:
        conn.execute(
            f"CREATE DATABASE \"{name}\" ENCODING 'SQL_ASCII' "
            "LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
        )
    s = None
    try:
        admin_dsn = with_db(admin, name)
        m = MemoryStore(admin_dsn)
        try:
            m.apply_all_migrations(admin_dsn=admin_dsn)
        finally:
            m.close()
        s = MemoryStore(with_db(runtime, name))
        with s._get_pool().connection() as conn:
            with conn.cursor() as cur:
                # More bad rows than the breaker threshold below: they must
                # not be mistaken for an endpoint outage.
                for i in range(4):
                    cur.execute(
                        "INSERT INTO memory_entries (agent_identity, target, content, "
                        "embedding, metadata) VALUES ('enc', 'memory', E'caf" + chr(92) + "351 note %d', " % i +
                        "NULL, '{}'::jsonb)"
                    )
                cur.execute(
                    "INSERT INTO memory_entries (agent_identity, target, content, "
                    "embedding, metadata) VALUES ('enc', 'memory', 'a good note', "
                    "NULL, '{}'::jsonb)"
                )
                conn.commit()
        seen = []

        def _embed_fn(text):
            seen.append(text.decode() if isinstance(text, bytes) else text)
            return [0.1] * 768

        report = s.backfill_null_embeddings(
            embed_fn=_embed_fn, tables=["memory_entries"], max_consecutive_failures=2,
        )["memory_entries"]
        assert [x for x in seen if x != "dimension probe"] == ["a good note"]
        assert report["skipped_changed"] == 0
        assert report["failed"] == 4
        assert report["succeeded"] == 1
        assert "note" not in report
    finally:
        if s is not None:
            s.close()
        with psycopg.connect(maint, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


# --- P2STORE-2: non-finite / non-numeric embedding values --------------------

def _serve(vec_json: str):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = (
                '{"data":[{"embedding":%s}],"embedding":%s,"embeddings":[%s]}'
                % (vec_json, vec_json, vec_json)
            ).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "null", '"x"', "true", "1" + "0" * 400])
def test_embed_rejects_non_finite_or_non_numeric_elements(bad):
    vals = ["0.5"] * 767 + [bad]
    srv = _serve("[" + ",".join(vals) + "]")
    try:
        with pytest.raises(EmbeddingError) as ei:
            embed("hello", base_url=f"http://127.0.0.1:{srv.server_port}")
        assert not isinstance(ei.value, EmbeddingDimensionError)
    finally:
        srv.shutdown()
        srv.server_close()


def test_embed_accepts_finite_ints_and_floats():
    srv = _serve(json.dumps([1] * 384 + [0.25] * 384))
    try:
        vec = embed("hello", base_url=f"http://127.0.0.1:{srv.server_port}")
        assert len(vec) == 768
    finally:
        srv.shutdown()
        srv.server_close()


# --- P2STORE-3A: remap_identity exact-equality guard -------------------------

def test_remap_padded_identity_to_trimmed_works(store):
    s, agent = store
    old, new = agent + "-pad ", agent + "-pad"
    s.add(agent_identity=old, target="memory", content="padded note")
    dry = s.remap_identity(old_identity=old, new_identity=new, dry_run=True)
    assert dry["memory_entries"]["moved"] == 1
    assert s.count(agent_identity=old) == 1
    s.remap_identity(old_identity=old, new_identity=new, dry_run=False)
    assert s.count(agent_identity=old) == 0
    assert s.count(agent_identity=new) == 1


def test_remap_identical_identity_still_refused(store):
    s, agent = store
    with pytest.raises(ValueError):
        s.remap_identity(old_identity=agent + "x", new_identity=agent + "x", dry_run=True)


# --- P2STORE-3B: replace() onto already-present content ----------------------

@pytest.mark.parametrize("mode", ["like", "exact"])
def test_replace_into_existing_content_drops_stale_row(store, mode):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="entry A")
    s.add(agent_identity=agent, target="memory", content="entry B")
    kwargs = {"exact_content": "entry A"} if mode == "exact" else {"old_text": "entry A"}
    n = s.replace(agent_identity=agent, target="memory", new_content="entry B", **kwargs)
    assert n == 1
    assert [r["content"] for r in s.list_entries(agent_identity=agent)] == ["entry B"]
    # Connection is still usable and normal replace still works afterwards.
    assert s.replace(agent_identity=agent, target="memory", old_text="entry B",
                     new_content="entry C") == 1


# --- P2OPS-1: has-text predicate must not depend on the locale ---------------

def _scratch_db(encoding, ctype):
    """Context manager: throwaway database with the given encoding/LC_CTYPE."""
    import contextlib

    import psycopg
    admin, runtime = os.environ.get("PG_TEST_ADMIN_DSN"), os.environ.get("PG_TEST_DSN")
    if not (admin and runtime):
        pytest.skip("PG_TEST_ADMIN_DSN / PG_TEST_DSN not set")

    def with_db(dsn, name):
        return re.sub(r"dbname=\S+", f"dbname={name}", dsn)

    @contextlib.contextmanager
    def _cm():
        name = "pytest_p2ops_" + os.urandom(3).hex()
        maint = with_db(admin, "postgres")
        with psycopg.connect(maint, autocommit=True) as conn:
            conn.execute(
                f"CREATE DATABASE \"{name}\" ENCODING '{encoding}' "
                f"LC_COLLATE 'C' LC_CTYPE '{ctype}' TEMPLATE template0"
            )
        s = None
        try:
            admin_dsn = with_db(admin, name)
            m = MemoryStore(admin_dsn)
            try:
                m.apply_all_migrations(admin_dsn=admin_dsn)
            finally:
                m.close()
            s = MemoryStore(with_db(runtime, name))
            yield s
        finally:
            if s is not None:
                s.close()
            with psycopg.connect(maint, autocommit=True) as conn:
                conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    return _cm()


def _predicate_mismatches(s, pred):
    """Code points where the SQL predicate disagrees with str.strip()."""
    # Surrogates (U+D800-U+DFFF) are not valid UTF8; chr(0) is not allowed.
    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT g FROM generate_series(1, 1114111) g "
                "WHERE g NOT BETWEEN 55296 AND 57343 "
                "AND (" + pred.replace("content", "chr(g)") + ") IS DISTINCT FROM "
                "(g NOT IN (" + ",".join(
                    str(c) for c in range(1, 0x110000)
                    if not (0xD800 <= c <= 0xDFFF) and not chr(c).strip()
                ) + "))"
            )
            return [r[0] for r in cur.fetchall()]


@pytest.mark.parametrize("ctype", ["C", "en_US.utf8"])
def test_has_text_predicate_matches_str_strip_on_every_code_point(ctype):
    from hermes_pgvector.store import _HAS_TEXT_SQL_UTF8
    import psycopg
    try:
        cm = _scratch_db("UTF8", ctype)
        with cm as s:
            assert s._has_text_sql() == _HAS_TEXT_SQL_UTF8
            assert _predicate_mismatches(s, _HAS_TEXT_SQL_UTF8) == []
    except psycopg.errors.InvalidParameterValue:
        pytest.skip(f"locale {ctype} not available in the test server")


def test_backfill_counts_unicode_blank_rows_unembeddable_on_c_ctype_utf8_db():
    with _scratch_db("UTF8", "C") as s:
        with s._get_pool().connection() as conn:
            with conn.cursor() as cur:
                blanks = [chr(0x3000), chr(0x2003), chr(0x2003) + chr(0x3000), chr(0x1680), chr(0x2028)]
                for i, blank in enumerate(blanks):
                    cur.execute(
                        "INSERT INTO memory_entries (agent_identity, target, content, "
                        "embedding, metadata) VALUES (%s, 'memory', %s, NULL, '{}'::jsonb)",
                        (f"blank{i}", blank),
                    )
                cur.execute(
                    "INSERT INTO memory_entries (agent_identity, target, content, "
                    "embedding, metadata) VALUES ('real', 'memory', 'real note', NULL, '{}'::jsonb)"
                )
                conn.commit()
        rep = s.backfill_null_embeddings(
            embed_fn=lambda t: [0.1] * 768, tables=["memory_entries"],
        )["memory_entries"]
        assert rep["unembeddable"] == 5
        assert rep["remaining"] == 0
        assert rep["succeeded"] == 1 and rep["failed"] == 0
