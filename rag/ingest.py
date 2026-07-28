import os
import nltk
from dataclasses import dataclass
from typing import List

# Ensure both 'punkt' and 'punkt_tab' tokenizer resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


def load_documents(folder: str) -> List[dict]:
    """Load every .txt file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    if not os.path.exists(folder):
        return docs
        
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        title = os.path.splitext(filename)[0].replace("_", " ").title()
        docs.append({"title": title, "text": text})
    return docs


def chunk_text_sentence_aware(text: str, max_words_per_chunk: int = 100) -> List[str]:
    """Group full sentences together up to a word count threshold."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > max_words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_chunk_records(docs: List[dict], max_words_per_chunk: int = 100) -> List[Chunk]:
    """Turn loaded documents into Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text_sentence_aware(doc["text"], max_words_per_chunk=max_words_per_chunk)
        for i, piece in enumerate(pieces):
            records.append(Chunk(chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], text=piece))
    return records