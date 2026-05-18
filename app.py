import os
import tempfile
from pathlib import Path
import random

import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv(".env.example")

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


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agent AG", page_icon="📄", layout="centered")
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

# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

chain, retriever = build_chain(uploaded.read(), uploaded.name)

# Chat history
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
            answer = chain.invoke(question)
            docs = retriever.invoke(question)
            pages = sorted({doc.metadata.get("page", "?") for doc in docs})

        st.markdown(answer)
        st.caption(f"Sources: pages {pages}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages,
    })
