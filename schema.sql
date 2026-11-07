CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    doc          text NOT NULL,
    content      text NOT NULL,
    content_hash text NOT NULL,
    embedding    vector(384) NOT NULL,          -- bge-small-en-v1.5, L2-normalized
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (doc, content_hash)                  -- re-ingesting unchanged text is a no-op
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

-- Document-level access control: who may read a chunk. '{}' (no one) until ingest sets it from docs/access.json.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS groups text[] NOT NULL DEFAULT '{}';
CREATE INDEX IF NOT EXISTS chunks_groups_idx ON chunks USING gin (groups);

-- Semantic answer cache: a new question reuses a verified answer when it means the same thing
-- (embedding candidate + NLI paraphrase check + identical numbers), for the same caller groups and corpus.
CREATE TABLE IF NOT EXISTS answer_cache (
    id         bigserial PRIMARY KEY,
    question   text NOT NULL,
    embedding  vector(384) NOT NULL,
    groups     text[] NOT NULL,                 -- sorted caller groups: a public caller never sees an engineering answer
    corpus     text NOT NULL,                   -- corpus version at answer time; old versions are deleted on ingest
    result     jsonb NOT NULL,
    hits       int NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS answer_cache_embedding_idx ON answer_cache USING hnsw (embedding vector_cosine_ops);
