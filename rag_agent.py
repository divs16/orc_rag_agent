#!/usr/bin/env python3
"""RAG agent for Oracle Recruiting Cloud knowledge base (Excel + transcript)."""

import os
import re
import sys
import pickle
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from google import genai
from google.genai import types

EXCEL_FILE = "ORC_Test_Cases_Detailed(Sheet1).xlsx"
TXT_FILE = "orc_transcript_knowledge.txt"
CACHE_FILE = ".embeddings_cache.pkl"
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"
TOP_K = 6


def load_excel_chunks(path: str) -> list[dict]:
    df = pd.read_excel(path, sheet_name=0)
    df = df.dropna(subset=["Test Case ID"])

    chunks = []
    for _, row in df.iterrows():
        parts = []
        if pd.notna(row.get("Module")):
            parts.append(f"Module: {row['Module']}")
        if pd.notna(row.get("Test Case Name")):
            parts.append(f"Test Case: {row['Test Case Name']}")
        if pd.notna(row.get("Preconditions")):
            parts.append(f"Preconditions: {row['Preconditions']}")
        if pd.notna(row.get("Test Steps")):
            parts.append(f"Test Steps: {row['Test Steps']}")
        if pd.notna(row.get("Expected Result")):
            parts.append(f"Expected Result: {row['Expected Result']}")
        if pd.notna(row.get("Remarks / Notes")):
            parts.append(f"Notes: {row['Remarks / Notes']}")

        text = "\n".join(parts)
        chunks.append({
            "id": row["Test Case ID"],
            "label": f"{row['Test Case ID']} | {row.get('Module', '')}",
            "name": row.get("Test Case Name", ""),
            "source": "test_cases",
            "text": text,
        })
    return chunks


def load_txt_chunks(path: str) -> list[dict]:
    raw = Path(path).read_text()
    # Split on === SECTION NAME === headers
    parts = re.split(r"(===\s*.+?\s*===)", raw)

    chunks = []
    section_name = "General"
    for part in parts:
        header_match = re.match(r"===\s*(.+?)\s*===", part)
        if header_match:
            section_name = header_match.group(1).strip().title()
            continue
        lines = [l.strip() for l in part.strip().splitlines() if l.strip()]
        if not lines:
            continue
        text = f"Topic: {section_name}\n" + "\n".join(lines)
        slug = re.sub(r"[^a-z0-9]+", "-", section_name.lower()).strip("-")
        chunks.append({
            "id": f"KB-{slug}",
            "label": f"Knowledge Base | {section_name}",
            "name": section_name,
            "source": "transcript",
            "text": text,
        })
    return chunks


def embed_texts(client: genai.Client, texts: list[str]) -> np.ndarray:
    # Gemini embedding API accepts up to 100 texts per batch
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
        )
        all_embeddings.extend([e.values for e in result.embeddings])
    return np.array(all_embeddings, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    c = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-9)
    return c @ q


def source_fingerprint(paths: list[str]) -> dict:
    """Map each path to its mtime so we can detect file changes."""
    return {p: Path(p).stat().st_mtime for p in paths if Path(p).exists()}


def load_or_build_index(client: genai.Client, excel_path: str, txt_path: str):
    cache = Path(CACHE_FILE)
    current_fp = source_fingerprint([excel_path, txt_path])

    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data.get("fingerprint") == current_fp:
            print("Loading cached embeddings...")
            return data["chunks"], data["embeddings"]
        print("Source files changed — rebuilding index...")

    chunks = []
    if Path(excel_path).exists():
        print(f"Reading {excel_path}...")
        excel_chunks = load_excel_chunks(excel_path)
        print(f"  {len(excel_chunks)} test cases loaded.")
        chunks.extend(excel_chunks)

    if Path(txt_path).exists():
        print(f"Reading {txt_path}...")
        txt_chunks = load_txt_chunks(txt_path)
        print(f"  {len(txt_chunks)} knowledge base sections loaded.")
        chunks.extend(txt_chunks)

    print(f"Generating embeddings for {len(chunks)} total chunks...")
    embeddings = embed_texts(client, [c["text"] for c in chunks])

    with open(cache, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings, "fingerprint": current_fp}, f)

    print("Embeddings cached.\n")
    return chunks, embeddings


def retrieve(query: str, client: genai.Client, chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=[query])
    query_vec = np.array(result.embeddings[0].values, dtype=np.float32)
    scores = cosine_similarity(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in top_indices]


def answer(query: str, context_chunks: list[dict], client: genai.Client) -> str:
    context = "\n\n---\n\n".join(
        f"[{c['label']}] {c['name']}\n{c['text']}" for c in context_chunks
    )
    prompt = f"""You are an expert on Oracle Recruiting Cloud (ORC) workflows and processes.
You have access to two knowledge sources:
1. Structured test cases (source: test_cases) — step-by-step test procedures with preconditions and expected results.
2. Transcript knowledge base (source: transcript) — business rules, policies, and configuration guidelines extracted from walkthroughs.

Use the retrieved excerpts below to answer the question. Synthesize across both sources when relevant.
Reference test case IDs (e.g. ORC-TC-001.1) or knowledge base topics when citing specific information.
If the answer is not covered by the context, say so honestly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text


def load_dotenv():
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set the GEMINI_API_KEY environment variable.")
        sys.exit(1)

    excel_path = sys.argv[1] if len(sys.argv) > 1 else EXCEL_FILE
    txt_path = sys.argv[2] if len(sys.argv) > 2 else TXT_FILE

    if not Path(excel_path).exists() and not Path(txt_path).exists():
        print(f"Error: neither '{excel_path}' nor '{txt_path}' found.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    chunks, embeddings = load_or_build_index(client, excel_path, txt_path)

    print("Oracle Recruiting Cloud RAG Agent")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        top_chunks = retrieve(query, client, chunks, embeddings)
        reply = answer(query, top_chunks, client)

        print(f"\nAgent: {textwrap.fill(reply, width=100, subsequent_indent='        ')}\n")


if __name__ == "__main__":
    main()
