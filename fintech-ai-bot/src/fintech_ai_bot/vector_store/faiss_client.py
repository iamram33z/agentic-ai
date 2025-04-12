import faiss
import numpy as np
import requests
import json
from typing import List, Dict, Optional
from pathlib import Path
import time

from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger

logger = get_logger(__name__)

class FAISSClient:
    """Client for managing FAISS index and document storage."""

    def __init__(self):
        self.index_path: Path = settings.faiss_index_path
        self.documents_path: Path = settings.faiss_docs_path
        self.dimension: int = settings.embedding_dimension
        self.model_name: str = settings.embedding_model_name
        self.hf_api_key: Optional[str] = settings.hf_api_key
        self.embedding_api_url: str = str(settings.embedding_api_url) # Convert HttpUrl
        self.embedding_timeout: int = settings.embedding_request_timeout
        self.k: int = settings.vector_search_k

        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self._initialize_store()

    def _initialize_store(self):
        """Loads or creates the FAISS index and documents."""
        try:
            # Ensure directories exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            if self.index_path.exists() and self.documents_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents and index with {self.index.ntotal} vectors.")
                # Sanity check
                if self.index.ntotal != len(self.documents):
                    logger.warning(f"Index size ({self.index.ntotal}) mismatch with document count ({len(self.documents)}). Re-indexing might be needed.")
                    # Decide on recovery strategy - e.g., rebuild index or log error
            else:
                logger.info("Creating new FAISS index.")
                # Using IndexFlatL2 as before, consider IndexIVFFlat for larger datasets
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
                self._load_initial_documents() # Try loading initial data if index is new

        except Exception as e:
            logger.critical(f"FAISS index initialization failed: {e}", exc_info=True)
            # Fallback to an empty in-memory index if loading fails critically
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []

    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Gets embeddings from Hugging Face Inference API with retry."""
        if not self.hf_api_key:
            logger.error("Hugging Face API key not configured.")
            return None
        if not texts:
            return np.array([], dtype='float32').reshape(0, self.dimension)

        headers = {
            "Authorization": f"Bearer {self.hf_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        retries = 2
        delay = 2

        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    self.embedding_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.embedding_timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                embeddings = response.json()
                # Basic validation of embedding structure/type
                if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                     # Convert and normalize (optional but recommended for cosine similarity if using IndexFlatIP)
                    embeddings_array = np.array(embeddings, dtype='float32')
                    # faiss.normalize_L2(embeddings_array) # Uncomment if using IndexFlatIP
                    return embeddings_array
                else:
                    logger.error(f"Unexpected embedding format received: {type(embeddings)}")
                    return None
            except requests.exceptions.Timeout:
                logger.warning(f"Embedding request timed out (Attempt {attempt + 1}).")
            except requests.exceptions.RequestException as e:
                logger.error(f"Embedding request failed (Attempt {attempt + 1}): {e}")
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response from embedding API (Attempt {attempt + 1})")

            if attempt < retries:
                logger.info(f"Retrying embedding request in {delay}s...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logger.error("Failed to get embeddings after multiple retries.")
                return None
        return None # Should not be reached

    def _save_store(self):
        """Saves the index and documents to disk."""
        if self.index is None:
            logger.error("Cannot save, FAISS index is not initialized.")
            return
        try:
            logger.info(f"Saving FAISS index to {self.index_path} ({self.index.ntotal} vectors)")
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saving {len(self.documents)} documents to {self.documents_path}")
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
            logger.info("FAISS index and documents saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index or documents: {e}", exc_info=True)

    def _load_initial_documents(self):
        """Loads and indexes documents from data/policies and data/products."""
        # Simplified - replace with actual PDF/CSV parsing logic
        initial_docs = []
        try:
            # Placeholder: Load from policies dir
            if settings.policies_dir and settings.policies_dir.exists():
                for policy_file in settings.policies_dir.glob("*.pdf"): # Example
                     # Add actual PDF text extraction here
                    text = f"Content from {policy_file.name}. Replace with real extraction."
                    initial_docs.append({'text': text, 'source': policy_file.name, 'type': 'policy'})
                logger.info(f"Found {len(initial_docs)} potential policy documents.")

            # Placeholder: Load from products dir
            if settings.products_dir and settings.products_dir.exists():
                 for product_file in settings.products_dir.glob("*.csv"): # Example
                    # Add actual CSV parsing here
                    text = f"Product updates from {product_file.name}. Replace with real parsing."
                    initial_docs.append({'text': text, 'source': product_file.name, 'type': 'product_update'})
                 logger.info(f"Found {len(initial_docs)} total potential documents.")

            if initial_docs:
                self.add_documents(initial_docs) # This will also save the index

        except Exception as e:
            logger.error(f"Failed to load initial documents: {e}", exc_info=True)


    def add_documents(self, documents: List[Dict[str, str]]):
        """Adds new documents, gets embeddings, updates index, and saves."""
        if not documents or self.index is None:
            return
        if not all('text' in doc for doc in documents):
             logger.error("Cannot add documents: 'text' field is missing in one or more documents.")
             return

        logger.info(f"Adding {len(documents)} new documents to FAISS store.")
        texts = [doc['text'] for doc in documents]
        try:
            embeddings = self._get_embeddings(texts)
            if embeddings is None or embeddings.shape[0] != len(texts):
                 logger.error("Failed to get valid embeddings for all documents. Aborting add operation.")
                 return

            start_index = self.index.ntotal
            self.index.add(embeddings)
            # Add corresponding metadata
            doc_metadata = []
            for i, doc in enumerate(documents):
                 doc_metadata.append({
                     "text": doc['text'], # Store full text or summary? Decide based on needs.
                     "source": doc.get('source', 'Unknown'),
                     "type": doc.get('type', 'document'),
                     "index_id": start_index + i # Link to the vector index position
                 })
            self.documents.extend(doc_metadata)

            # Persist changes
            self._save_store()

        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}", exc_info=True)
            # Consider rollback or cleanup if partial add occurred


    def search(self, query: str) -> Optional[List[Dict]]:
        """Searches the index for relevant documents."""
        if not query or self.index is None or self.index.ntotal == 0:
            return None

        logger.info(f"Performing FAISS search for query: '{query[:100]}...'")
        try:
            start_time = time.monotonic()
            query_embedding = self._get_embeddings([query])

            if query_embedding is None or query_embedding.shape[0] == 0:
                logger.error("Could not generate embedding for the search query.")
                return None

            distances, indices = self.index.search(query_embedding, self.k)
            duration = time.monotonic() - start_time
            logger.info(f"FAISS search completed in {duration:.4f}s")

            results = []
            if len(indices) > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.documents): # Check index validity
                        doc = self.documents[idx]
                        # Simple heuristic for relevance based on L2 distance (lower is better)
                        # You might want to normalize scores or use a threshold
                        score = float(distances[0][i])
                        results.append({
                            'score': score,
                            'text': doc['text'][:settings.max_doc_tokens_in_prompt * 4] + ('...' if len(doc['text']) > settings.max_doc_tokens_in_prompt * 4 else ''), # Truncate based on config
                            'source': doc.get('source', 'Unknown'),
                            'type': doc.get('type', 'document')
                        })
                    else:
                         logger.warning(f"Search returned invalid index: {idx}")

            return results if results else None

        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            return None

# Example Usage:
# faiss_client = FAISSClient()
# search_results = faiss_client.search("What is the investment policy?")
# if search_results:
#     for result in search_results:
#         print(f"Source: {result['source']}, Score: {result['score']:.2f}")
#         print(result['text'])
#         print("---")