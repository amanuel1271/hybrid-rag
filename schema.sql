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
