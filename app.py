import os
import tempfile
import random
import uuid
import time

import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
    load_dotenv(".env.example")
except ImportError:
    pass

# Streamlit Cloud secrets → env vars (no-op locally if secrets.toml absent)
try:
    for key in ("ANTHROPIC_API_KEY", "LANGCHAIN_API_KEY", "POSTHOG_API_KEY"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

# ── LangSmith: enable tracing when key is present ────────────────────────────
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ask-the-doc")

# ── PostHog ───────────────────────────────────────────────────────────────────
try:
    import posthog as _ph
    _posthog_key = os.environ.get("POSTHOG_API_KEY", "")
    if _posthog_key:
        _ph.project_api_key = _posthog_key
        _ph.host = "https://us.i.posthog.com"
    else:
        _ph.disabled = True
    _posthog_ok = True
except ImportError:
    _posthog_key = ""
    _posthog_ok = False


def track(event: str, **props):
    if not _posthog_ok or not _posthog_key:
        return
    _ph.capture(st.session_state["session_id"], event, properties=props)


# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAG_PROMPT = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    'If the answer isn\'t in the context, say "I don\'t know."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\nAnswer:"
)

mission_messages = [
    "Agent AG is on the case…",
    "Agent AG is gathering intel…",
    "Agent AG is cracking the code…",
    "Agent AG is infiltrating the data…",
    "Agent AG is connecting the dots…",
    "Agent AG's mission: activated…",
]


@st.cache_resource(show_spinner="Building vector store…")
def build_chain(pdf_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_bytes)
        tmp_path = f.name

    pages = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=None)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agent AG", page_icon="📄", layout="centered")

# Stable session ID for PostHog (one per browser session)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
    track("page_view")

st.title("Agent AG")
st.caption("Upload a PDF and ask questions about its content — powered by Claude + RAG")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY not found. Add it to a `.env` file in this folder.")
    st.stop()

# ── Sidebar: upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded:
        st.success(f"{uploaded.name} loaded")
        st.caption(f"{uploaded.size // 1024} KB")

        if uploaded.name != st.session_state.get("last_uploaded"):
            st.session_state["last_uploaded"] = uploaded.name
            track("pdf_uploaded", filename=uploaded.name, size_kb=uploaded.size // 1024)

# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Read bytes once; cache in session state so rerenders don't get empty bytes
if st.session_state.get("pdf_name") != uploaded.name:
    st.session_state["pdf_bytes"] = uploaded.read()
    st.session_state["pdf_name"] = uploaded.name

chain, retriever = build_chain(st.session_state["pdf_bytes"], uploaded.name)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("pages"):
            st.caption(f"Sources: pages {msg['pages']}")

if question := st.chat_input("Ask a question about your PDF…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(random.choice(mission_messages)):
            t0 = time.time()
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            latency_ms = int((time.time() - t0) * 1000)
            pages = sorted({doc.metadata.get("page", "?") for doc in docs})

        st.markdown(answer)
        st.caption(f"Sources: pages {pages}")

    track(
        "question_asked",
        question=question,
        source_pages=pages,
        num_sources=len(docs),
        latency_ms=latency_ms,
        pdf=uploaded.name,
        turn=len(st.session_state.messages) // 2,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages,
    })
