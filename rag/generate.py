import os
from typing import List, Tuple
from google import genai
from .ingest import Chunk

# Your Google AI Studio API Key
API_KEY = "AQ.Ab8RN6KeHlmTPgfw7KwJSvLhU1C4ZiacGfaUMx353WqzT_r2nQ"


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]], similarity_threshold: float = 0.35) -> str:
    valid_chunks = [(c, score) for c, score in retrieved if score >= similarity_threshold]

    if not valid_chunks:
        return "I cannot find relevant information in the provided document collection to answer your question."

    lines = [f"Top passages related to: \u201c{query}\u201d\n"]
    for chunk, score in valid_chunks:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], similarity_threshold: float = 0.25) -> str:
    valid_chunks = [c for c, score in retrieved if score >= similarity_threshold]

    if not valid_chunks:
        return "I cannot find relevant information in the provided document collection to answer your question."

    context = "\n\n".join(f"Source Document: [{c.doc_title}]\nContent: {c.text}" for c in valid_chunks)

    prompt = (
        "You are an expert AI search and recommendation assistant for Manga, Manhwa, Webtoons, and Visual Novels.\n"
        "Answer the user's question and provide helpful recommendations using the provided source context.\n"
        "Treat 'manga', 'manhwa', and 'webtoons' flexibly when recommending titles unless the user explicitly asks to distinguish them.\n"
        "If the answer cannot be determined from the context at all, state clearly that you do not have enough information.\n"
        "Always cite your sources in the answer using brackets like [Doc Title].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    try:
        client = genai.Client(api_key=API_KEY)
        # Using the official auto-updating flash alias supported by the google-genai SDK
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def generate_answer(query: str, retrieved: List[Tuple[Chunk, float]], mode: str = "extractive") -> str:
    if mode == "llm":
        return llm_answer(query, retrieved)
    return extractive_answer(query, retrieved)