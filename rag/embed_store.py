import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from .ingest import Chunk


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the dense embedding model using sentence-transformers."""
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = np.array([])

    def build(self, chunks: List[Chunk]):
        """Encode document chunks into dense vector embeddings."""
        self.chunks = chunks
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        # Generate dense embeddings (returns normalized 384-dim numpy array)
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Encode query and compute cosine similarity against indexed chunks."""
        if self.embeddings.size == 0 or not self.chunks:
            return []

        # Encode query into a normalized vector
        query_embedding = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]

        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.embeddings, query_embedding)

        # Get top-k indices ranked by highest similarity score
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]