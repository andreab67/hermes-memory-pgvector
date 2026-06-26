import os
import pytest
import psycopg
from hexus.store import MemoryStore

# Skip the whole module if there's no DSN to talk to.
pytestmark = pytest.mark.skipif(
    not os.environ.get("PG_TEST_DSN"),
    reason="PG_TEST_DSN not set — live DB tests skipped",
)


@pytest.fixture
def clean_db():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS delegations CASCADE;")
            cur.execute("DROP TABLE IF EXISTS conversations CASCADE;")
            cur.execute("DROP TABLE IF EXISTS memory_entries CASCADE;")
            
    # Re-apply migrations using apply_migration_as_admin
    s = MemoryStore(dsn)
    s.apply_migration_as_admin(admin_dsn=dsn)
    s.ensure_schema()
    return s


def test_quantization_float16_adaptation(clean_db, monkeypatch):
    """Verify that setting HEXUS_VECTOR_PRECISION=float16 alters columns to halfvec and creates halfvec indexes."""
    dsn = os.environ.get("PG_TEST_DSN")
    monkeypatch.setenv("HEXUS_VECTOR_PRECISION", "float16")
    
    # Instantiate new store with float16 precision
    store = MemoryStore(dsn)
    store.ensure_schema() # This calls adapt_vector_precision()
    
    # 1. Verify column type is halfvec(384)
    with store._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pg_catalog.format_type(atttypid, atttypmod)
                FROM pg_catalog.pg_attribute
                WHERE attrelid = 'memory_entries'::regclass
                  AND attname = 'embedding';
            """)
            col_type = cur.fetchone()[0]
            assert col_type == "halfvec(384)"
            
            cur.execute("""
                SELECT pg_catalog.format_type(atttypid, atttypmod)
                FROM pg_catalog.pg_attribute
                WHERE attrelid = 'conversations'::regclass
                  AND attname = 'embedding';
            """)
            conv_col_type = cur.fetchone()[0]
            assert conv_col_type == "halfvec(384)"
            
    # 2. Verify we can insert and search successfully
    agent = "test-agent-float16"
    store.add(agent_identity=agent, target="memory", content="Python fp16 vector test")
    
    # Retrieve query
    emb = [0.1] * 384
    res = store.search(query_embedding=emb, agent_identity=agent, limit=1)
    assert len(res) == 1
    assert res[0]["content"] == "Python fp16 vector test"


def test_quantization_binary_adaptation(clean_db, monkeypatch):
    """Verify that setting HEXUS_VECTOR_PRECISION=binary drops cosine indexes and creates binary indexes."""
    dsn = os.environ.get("PG_TEST_DSN")
    monkeypatch.setenv("HEXUS_VECTOR_PRECISION", "binary")
    
    # Instantiate new store with binary precision
    store = MemoryStore(dsn)
    store.ensure_schema() # This calls adapt_vector_precision()
    
    # 1. Verify binary index exists
    with store._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'memory_entries' 
                  AND indexname = 'ix_memory_entries_embedding_binary_hnsw';
            """)
            assert cur.fetchone() is not None

            # Verify standard cosine index does NOT exist
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'memory_entries' 
                  AND indexname = 'ix_memory_entries_embedding_hnsw';
            """)
            assert cur.fetchone() is None
            
    # 2. Verify we can insert and search (using two-stage search) successfully
    agent = "test-agent-binary"
    store.add(agent_identity=agent, target="memory", content="Python binary vector test")
    
    # Retrieve query
    emb = [0.1] * 384
    res = store.search(query_embedding=emb, agent_identity=agent, limit=1)
    assert len(res) == 1
    assert res[0]["content"] == "Python binary vector test"
