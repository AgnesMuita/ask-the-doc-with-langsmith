# Agent AG — Ask your PDF

Upload any PDF and ask questions about it in plain English. Powered by Claude AI and a local RAG pipeline.

**[Live demo →][(https://ask-the-doc.streamlit.app](https://ask-the-doc-with-langsmith.streamlit.app/)**

---

## How it works

```
PDF → split into chunks → embed locally → store in Chroma
                                                  │
Your question ──── similarity search ────► top 4 chunks
                                                  │
                                         Claude Haiku answers
```

1. **Ingest** — the PDF is split into 1000-character chunks with overlap
2. **Embed** — each chunk is converted to a vector using `all-MiniLM-L6-v2` (runs locally, no API call)
3. **Retrieve** — your question is embedded and the 4 closest chunks are found
4. **Generate** — Claude Haiku answers using only those chunks as context

## Stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| LLM | Claude Haiku (`claude-haiku-4-5`) via `langchain-anthropic` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | Chroma (in-memory) |
| Orchestration | LangChain LCEL |
| Tracing | LangSmith |
| Analytics | PostHog |

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/ask-the-doc
cd ask-the-doc
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=...       # optional — enables LangSmith tracing
POSTHOG_API_KEY=...         # optional — enables analytics
```

```bash
python -m streamlit run app.py
```

## Deploy

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Secrets are set via the app dashboard — no `.env` file needed in production.
