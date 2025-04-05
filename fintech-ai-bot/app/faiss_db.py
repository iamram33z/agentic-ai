import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from typing import List


class FAISSDB:
    def __init__(self):
        self.index_path = os.getenv("FAISS_INDEX_PATH")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        else:
            return faiss.IndexFlatL2(384)  # 384-dim embeddings

    def search(self, query: str, k: int = 3) -> List[str]:
        # Embed query
        query_embedding = self.model.encode([query])

        # FAISS search
        distances, indices = self.index.search(query_embedding, k)

        # Return top matches (mock)
        return ["Policy A", "Update B", "Guideline C"]  # Replace with real data