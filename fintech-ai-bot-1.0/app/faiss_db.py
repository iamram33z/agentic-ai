import faiss
import numpy as np
import os
import requests
from typing import List, Dict, Optional
import json
from utils import get_logger
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = get_logger("FAISSVectorStore")


class FAISSVectorStore:
    def __init__(self):
        # Configure paths using os
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.index_path = os.path.join(self.base_dir, "faiss", "faiss_index")
        self.documents_path = os.path.join(self.base_dir, "faiss", "documents.json")
        self.policies_dir = os.path.join(self.base_dir, "data", "policies")
        self.products_dir = os.path.join(self.base_dir, "data", "products")

        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.documents = []
        self._initialize_index()

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings with retry logic and timeout"""
        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}",
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=10
            )
            response.raise_for_status()
            return np.array(response.json(), dtype='float32')
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request failed: {str(e)}")
            raise

    def _initialize_index(self):
        """Initialize index with proper error handling"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

            if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
                logger.info("Loading existing FAISS index and documents")
                self.index = faiss.read_index(self.index_path)
                with open(self.documents_path, 'r') as f:
                    self.documents = json.load(f)
            else:
                logger.info("Creating new FAISS index")
                self.index = faiss.IndexFlatL2(self.dimension)
                self._load_initial_documents()

            logger.info(f"Index initialized with {self.index.ntotal} documents")
        except Exception as e:
            logger.critical(f"Index initialization failed: {str(e)}")
            # Fallback to empty index
            self.index = faiss.IndexFlatL2(self.dimension)

    def _load_initial_documents(self):
        """Load initial document set from data directory"""
        try:
            # Load company policies
            policy_path = os.path.join(self.policies_dir, "company_policies.pdf")
            if os.path.exists(policy_path):
                # In production, you would use PyPDF2 or similar to extract text
                documents = [{
                    'text': "Sample policy document content. In production, this would be extracted from the PDF.",
                    'source': "company_policies.pdf",
                    'type': 'policy'
                }]
                self.add_documents(documents)
                logger.info("Loaded initial policy documents")

            # Load product updates
            updates_path = os.path.join(self.products_dir, "latest_updates.csv")
            if os.path.exists(updates_path):
                # In production, you would parse the CSV
                documents = [{
                    'text': "Sample product updates. In production, this would be parsed from the CSV.",
                    'source': "latest_updates.csv",
                    'type': 'product_update'
                }]
                self.add_documents(documents)
                logger.info("Loaded initial product updates")

        except Exception as e:
            logger.error(f"Failed to load initial documents: {str(e)}")

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store with metadata"""
        if not documents:
            return

        try:
            texts = [doc['text'] for doc in documents]
            embeddings = self._get_embeddings(texts)

            # Add to index
            self.index.add(embeddings)

            # Add to documents store
            self.documents.extend(documents)

            # Persist changes
            self._save_index()

            logger.info(f"Added {len(documents)} new documents to index")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def _save_index(self):
        """Save index and documents to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.documents_path, 'w') as f:
                json.dump(self.documents, f)
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")

    def search(self, query: str, k: int = 3) -> Optional[str]:
        """Enhanced semantic search with metadata"""
        if not query or self.index.ntotal == 0:
            return None

        try:
            start_time = datetime.now()

            # Get query embedding
            query_embedding = self._get_embeddings([query])

            # Search index
            distances, indices = self.index.search(query_embedding, k)

            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:  # FAISS returns -1 for invalid indices
                    doc = self.documents[idx]
                    results.append({
                        'score': float(distances[0][i]),
                        'text': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text'],
                        'source': doc.get('source', 'Unknown'),
                        'type': doc.get('type', 'document')
                    })

            logger.info(f"Search completed in {(datetime.now() - start_time).total_seconds():.2f}s")

            if not results:
                return None

            # Format results as markdown
            return self._format_results(results)

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return None

    def _format_results(self, results: List[Dict]) -> str:
        """Format search results for display"""
        formatted = ["## 📚 Relevant Documents"]

        for i, res in enumerate(results, 1):
            formatted.append(f"""
### 📄 Document {i} ({res['type'].title()})
**Source**: {res['source']}  
**Relevance Score**: {res['score']:.2f}

{res['text']}
""")

        return "\n".join(formatted)


if __name__ == "__main__":
    # Enhanced test script
    store = FAISSVectorStore()

    test_queries = [
        "investment strategy",
        "risk management policies",
        "product updates"
    ]

    for query in test_queries:
        print(f"\n=== Testing query: '{query}' ===")
        results = store.search(query)
        print(results or "No results found")