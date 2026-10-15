-- ==============================================================================
-- Paper Mind Database Schema (PostgreSQL + pgvector for Supabase)
-- ==============================================================================

-- 1. Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Workspaces table: Isolates collections of papers and chat threads
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Papers table: Uploaded PDF documents scoped to a workspace
CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    upload_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_text TEXT NOT NULL,
    extraction_quality_report JSONB
);

-- 4. Chunks table: Text chunks with dense vector embeddings and metadata
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL CHECK (chunking_strategy IN ('fixed', 'section_aware')),
    section_label TEXT,
    page_number INT,
    embedding vector(384) -- 384 dimensions matching all-MiniLM-L6-v2
);

-- 5. Messages table: Chat history and agent reasoning traces
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reasoning_trace JSONB
);

-- ==============================================================================
-- Indices for Performance and Similarity Search
-- ==============================================================================

-- Relational filtering indices
CREATE INDEX IF NOT EXISTS idx_papers_workspace_id ON papers(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_id ON chunks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(chunking_strategy);
CREATE INDEX IF NOT EXISTS idx_messages_workspace_id ON messages(workspace_id);

-- Vector similarity index (HNSW for fast cosine distance search)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops);

-- ==============================================================================
-- Stored Procedure: Scoped Vector Similarity Search
-- ==============================================================================

CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(384),
    filter_workspace_id UUID,
    filter_strategy TEXT DEFAULT NULL,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    paper_id UUID,
    workspace_id UUID,
    chunk_text TEXT,
    chunking_strategy TEXT,
    section_label TEXT,
    page_number INT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.paper_id,
        c.workspace_id,
        c.chunk_text,
        c.chunking_strategy,
        c.section_label,
        c.page_number,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.workspace_id = filter_workspace_id
      AND (filter_strategy IS NULL OR c.chunking_strategy = filter_strategy)
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
