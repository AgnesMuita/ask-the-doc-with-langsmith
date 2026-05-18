"""
RAG demo: ask questions about a PDF using LangChain + Claude.

Usage:
    python rag_pdf.py                         # uses the bundled sample PDF
    python rag_pdf.py --pdf path/to/your.pdf  # use your own PDF
    python rag_pdf.py --query "What is this about?"
"""

import argparse
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")
load_dotenv(".env.example")   # fallback for dev

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

SAMPLE_PDF_URL = "https://www.irs.gov/pub/irs-pdf/fw9.pdf"
SAMPLE_PDF_PATH = Path(__file__).parent / "sample.pdf"

RAG_PROMPT = PromptTemplate.from_template(
    "Use only the context below to answer the question. "
    'If the answer isn\'t in the context, say "I don\'t know."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def download_sample_pdf() -> str:
    if not SAMPLE_PDF_PATH.exists():
        print("Downloading sample PDF (IRS W-9 form)...")
        urllib.request.urlretrieve(SAMPLE_PDF_URL, SAMPLE_PDF_PATH)
        print(f"Saved to {SAMPLE_PDF_PATH}")
    else:
        print(f"Using cached sample: {SAMPLE_PDF_PATH}")
    return str(SAMPLE_PDF_PATH)


def build_chain(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    print(f"  {len(pages)} pages loaded")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(pages)
    print(f"  {len(chunks)} chunks after splitting")

    print("Building vector store (local embeddings, no API call)...")
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
    print("Ready.\n")
    return chain, retriever


def ask(chain, retriever, question: str):
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    pages = sorted({doc.metadata.get("page", "?") for doc in docs})
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"   (sources: pages {pages})\n")


def main():
    parser = argparse.ArgumentParser(description="PDF RAG demo")
    parser.add_argument("--pdf", default=None, help="Path to a PDF file")
    parser.add_argument("--query", default=None, help="Single question (skips REPL)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Error: ANTHROPIC_API_KEY not found. Add it to .env or set it in your shell.")

    pdf_path = args.pdf or download_sample_pdf()
    chain, retriever = build_chain(pdf_path)

    if args.query:
        ask(chain, retriever, args.query)
        return

    print("Ask questions about the PDF (type 'quit' to exit):")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if question:
            ask(chain, retriever, question)


if __name__ == "__main__":
    main()
