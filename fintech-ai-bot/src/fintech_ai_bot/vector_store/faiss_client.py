# faiss_client.py (Corrected add_documents check)

import faiss
import numpy as np
import json
from typing import List, Dict, Optional, Generator, Tuple
from pathlib import Path
import time
import torch
from transformers import pipeline, AutoTokenizer, PreTrainedTokenizerBase
import logging
import math
import gc  # Garbage collection

# --- Logger Setup ---
try:
    from fintech_ai_bot.config import settings
    from fintech_ai_bot.utils import get_logger, log_execution_time, generate_error_html

    logger = get_logger(__name__)
except ImportError as import_err:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Could not import from fintech_ai_bot modules (Error: {import_err}). Using basic fallback logger and mock settings.")


    class MockSettings:
        faiss_index_path = Path("./faiss_index").resolve()
        faiss_docs_path = Path("./documents.json").resolve()
        embedding_dimension = 768
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_api_key = None
        vector_search_k = 3
        policies_dir = Path("./data/policies").resolve()
        products_dir = Path("./data/products").resolve()
        max_doc_chars_in_prompt = 25000
        log_dir = Path("./logs").resolve()
        chunk_size_tokens: int = 450
        chunk_overlap_tokens: int = 50
        embedding_batch_size: int = 32


    settings = MockSettings()


    def log_execution_time(func):
        return func  # Mock decorator


    def generate_error_html(msg, det=""):
        return f"ERROR: {msg} Details: {det}"  # Mock error func


# --- End Logger Setup ---


# --- Enhanced Text Chunking Helper ---
class TokenChunker:
    """Splits text into chunks based on token count using a Hugging Face tokenizer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, chunk_size: int, chunk_overlap: int):
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Tokenizer must be valid.")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("Chunk size positive int.")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("Chunk overlap non-negative int.")
        if chunk_overlap >= chunk_size:
            logger.warning(f"Overlap({chunk_overlap})>=Size({chunk_size}). Adjusting->{chunk_size // 3}.")
            chunk_overlap = chunk_size // 3

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        model_max_len = getattr(tokenizer, 'model_max_length', None)
        self.effective_chunk_size = chunk_size
        if model_max_len and model_max_len < chunk_size:
            self.effective_chunk_size = model_max_len
            logger.info(f"Chunk size adjusted to tokenizer max length: {self.effective_chunk_size}")
        logger.info(f"TokenChunker initialized: Size={self.effective_chunk_size}, Overlap={self.chunk_overlap}")

    def split_text(self, text: str) -> List[str]:
        """Splits text into chunks respecting token limits."""
        if not text or not isinstance(text, str):
            logger.warning("Empty/non-string for chunking.")
            return []

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.error(f"Tokenizer encode failed: {e}", exc_info=True)
            return [text] if 0 < len(text) < self.effective_chunk_size * 4 else []

        token_count = len(tokens)
        chunks = []
        start_token_idx = 0

        if token_count == 0:
            logger.warning("Zero tokens after encoding.")
            return []

        if token_count <= self.effective_chunk_size:
            return [self.tokenizer.decode(tokens, skip_special_tokens=True).strip()]

        step_size = self.effective_chunk_size - self.chunk_overlap
        if step_size <= 0:
            logger.error("Invalid Chunker state: Step <= 0.")
            return [self.tokenizer.decode(tokens[:self.effective_chunk_size], skip_special_tokens=True).strip()]

        while start_token_idx < token_count:
            end_token_idx = min(start_token_idx + self.effective_chunk_size, token_count)
            chunk_token_ids = tokens[start_token_idx:end_token_idx]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True).strip()

            if chunk_text:
                chunks.append(chunk_text)
            else:
                logger.debug(f"Skipping empty chunk: indices {start_token_idx}-{end_token_idx}.")

            start_token_idx += step_size
            if end_token_idx == token_count:
                break

        logger.debug(f"Split text into {len(chunks)} chunks.")
        return chunks


# --- FAISS Client Class ---
class FAISSClient:
    """Manages FAISS index using local embeddings, chunking, and pooling."""

    def __init__(self):
        self.index_path: Path = settings.faiss_index_path
        self.documents_path: Path = settings.faiss_docs_path
        self.dimension: int = settings.embedding_dimension
        self.model_name: str = settings.embedding_model_name
        self.hf_api_key: Optional[str] = settings.hf_api_key
        self.k: int = settings.vector_search_k
        self.chunk_size: int = settings.chunk_size_tokens
        self.chunk_overlap: int = settings.chunk_overlap_tokens
        self.embedding_batch_size: int = settings.embedding_batch_size

        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self.feature_extractor = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.chunker: Optional[TokenChunker] = None
        self.is_initialized: bool = False

        # --- Initialization ---
        logger.info("Initializing FAISSClient...")
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self._initialize_embedding_pipeline_and_tokenizer()

        if self.feature_extractor and self.tokenizer:
            try:
                self.chunker = TokenChunker(self.tokenizer, self.chunk_size, self.chunk_overlap)
                self._initialize_store()  # Calls _load_initial_documents if index created

                if self.index is not None:
                    self.is_initialized = True
                    logger.info("FAISSClient initialized successfully.")
                else:
                    logger.error("FAISS store initialization failed. Client unusable.")
            except ValueError as chunker_err:
                logger.critical(f"Failed TokenChunker init: {chunker_err}")
                raise
            except Exception as store_err:
                logger.critical(f"Failed store initialization: {store_err}", exc_info=True)
        else:
            logger.error("Pipeline/tokenizer failed init. Client unusable.")

        if not self.is_initialized:
            logger.warning("FAISSClient did not initialize successfully.")

    def _initialize_embedding_pipeline_and_tokenizer(self):
        """Initializes the Hugging Face pipeline and tokenizer."""
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing '{self.model_name}' on {device}.")
            print(f"Device set to use {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_api_key)
            logger.info("Tokenizer initialized.")

            self.feature_extractor = pipeline(
                "feature-extraction",
                model=self.model_name,
                tokenizer=self.tokenizer,
                token=self.hf_api_key,
                device=device
            )

            try:
                _ = self.feature_extractor("test")
                logger.info("Pipeline initialized and tested.")
            except Exception as test_e:
                logger.warning(f"Pipeline test failed: {test_e}", exc_info=True)
        except Exception as e:
            logger.critical(f"Failed init: {e}", exc_info=True)
            self.feature_extractor = None
            self.tokenizer = None

    def _initialize_store(self):
        """Loads or creates the FAISS index and chunk document store."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            if self.index_path.exists() and self.documents_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                except Exception as e:
                    logger.critical(f"Failed read index: {e}", exc_info=True)
                    self.index = None
                    self.documents = []
                    logger.warning("Treating as non-existent.")
                    raise

                if self.index and self.index.d != self.dimension:
                    logger.critical(f"Index dim ({self.index.d}) != config ({self.dimension}). Delete files.")
                    raise ValueError("Index dim mismatch. Delete files.")

                logger.info(f"Loading existing chunk docs from {self.documents_path}")
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)

                logger.info(
                    f"Loaded {len(self.documents)} chunks & index ({self.index.ntotal if self.index else 'N/A'} vectors).")

                if self.index and self.index.ntotal != len(self.documents):
                    logger.warning(
                        f"Index count ({self.index.ntotal}) != doc count ({len(self.documents)}). Re-index suggested.")
            else:
                if self.index is None:
                    logger.info(f"Creating new FAISS index ({self.dimension}D) & store.")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.documents = []
                    self._load_initial_documents()
                else:
                    logger.warning("Index exists but docs file missing?")
        except ValueError as ve:
            logger.critical(str(ve))
            self.index = None
            self.documents = []
            raise ve
        except Exception as e:
            logger.critical(f"Store init failed: {e}", exc_info=True)
            logger.warning("FAISS has no index/docs.")
            self.index = None
            self.documents = []

    @log_execution_time
    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Gets embeddings for text chunks using batched processing and mean pooling."""
        if self.feature_extractor is None:
            logger.error("Pipeline not ready.")
            return None

        if not texts or not all(isinstance(t, str) for t in texts):
            logger.error(f"Invalid input type.")
            return None

        valid_texts = [t for t in texts if t.strip()]
        texts_were_empty = len(valid_texts) < len(texts)

        if not valid_texts:
            logger.warning("Input only empty strings.")
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        if texts_were_empty:
            logger.warning(f"Removed {len(texts) - len(valid_texts)} empty strings.")

        all_pooled = []
        total_texts = len(valid_texts)
        batch_size = self.embedding_batch_size
        num_batches = math.ceil(total_texts / batch_size)

        logger.info(f"Embedding {total_texts} chunks in {num_batches} batches (size {batch_size}).")

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, total_texts)
            batch_texts = valid_texts[batch_start:batch_end]

            logger.debug(f"Processing batch {i + 1}/{num_batches}...")
            start_batch_time = time.monotonic()

            try:
                output = self.feature_extractor(batch_texts)
                batch_pooled = []
                failed = False

                if len(output) != len(batch_texts):
                    logger.error(f"Batch {i + 1}: Output len!=input. Abort.")
                    return None

                for j, item_output in enumerate(output):
                    arr = None
                    try:
                        temp = np.array(item_output, dtype=np.float32)
                    except ValueError as ve:
                        logger.error(f"B{i + 1}, I{j + 1}: Array Err: {ve}. Skip.")
                        failed = True
                        break

                    if temp.ndim == 3 and temp.shape[0] == 1:
                        arr = temp.squeeze(axis=0)
                    elif temp.ndim == 2:
                        arr = temp
                    else:
                        logger.warning(f"B{i + 1}, I{j + 1}: Bad shape {temp.shape}. Skip.")
                        failed = True
                        break

                    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
                        logger.warning(
                            f"B{i + 1}, I{j + 1}: Invalid 2D arr (Shape:{arr.shape if arr is not None else 'N/A'}). Skip.")
                        failed = True
                        break

                    pooled = np.mean(arr, axis=0)

                    if pooled.ndim == 1 and pooled.shape[0] == self.dimension:
                        batch_pooled.append(pooled)
                    else:
                        logger.warning(f"B{i + 1}, I{j + 1}: Pooled shape {pooled.shape} invalid. Skip.")
                        failed = True
                        break

                if failed:
                    logger.error(f"Batch {i + 1} failed. Abort.")
                    return None

                all_pooled.extend(batch_pooled)
                batch_duration = time.monotonic() - start_batch_time
                logger.debug(f"Batch {i + 1} OK ({batch_duration:.2f}s).")
                del output, batch_pooled, arr, pooled, temp
                gc.collect()
            except IndexError as idx:
                logger.error(f"Batch {i + 1}: IndexError: {idx}", exc_info=True)
                return None
            except Exception as e:
                logger.error(f"Batch {i + 1}: Unexpected error: {e}", exc_info=True)
                return None

        if len(all_pooled) != total_texts:
            logger.error(f"Final count mismatch.")
            return None

        final_array = np.array(all_pooled, dtype=np.float32)

        if final_array.shape != (total_texts, self.dimension):
            logger.error(f"Final shape mismatch.")
            return None

        logger.info(f"Successfully embedded {total_texts} chunks.")
        return final_array

    def _save_store(self):
        """Saves the index and chunk documents to disk."""
        if self.index is None:
            logger.error("Cannot save, FAISS index object is None.")
            return

        if self.index.ntotal <= 0 and not self.documents:
            logger.info("Skipping save empty store.")
            try:
                with open(self.documents_path, 'w') as f:
                    json.dump([], f)
                logger.info("Wrote empty docs file.")
            except IOError as e:
                logger.error(f"Failed write empty docs file: {e}")
            return

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving index {self.index_path} ({self.index.ntotal} vectors)")
            faiss.write_index(self.index, str(self.index_path))

            logger.info(f"Saving {len(self.documents)} chunk docs {self.documents_path}")
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)

            logger.info("Store saved successfully.")
        except IOError as ioe:
            logger.error(f"IOError saving store: {ioe}", exc_info=True)
        except Exception as e:
            logger.error(f"Error saving store: {e}", exc_info=True)

    def _load_initial_documents(self):
        """Loads initial documents, extracts text, and calls add_documents."""
        if self.index is None:
            logger.warning("Cannot load initial documents: FAISS index object is None.")
            return

        initial_docs = []
        processed = set()

        if self.documents_path.exists() and self.documents_path.is_file():
            try:
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        processed.update(d.get('source') for d in data if isinstance(d, dict) and 'source' in d)
                        logger.info(f"Found {len(processed)} sources in existing chunks.")
            except Exception as e:
                logger.warning(f"Could not read/parse existing chunks file {self.documents_path}: {e}")

        def ext_pdf(fp: Path):
            try:
                import fitz
                doc = fitz.open(fp)
                txt = "".join(p.get_text() for p in doc).strip()
                doc.close()
                return txt or None
            except ImportError:
                logger.warning("PyMuPDF not installed.")
                return None
            except Exception as e:
                logger.error(f"PDF Error {fp.name}: {e}")
                return None

        def ext_csv(fp: Path):
            try:
                import pandas as pd
                df = pd.read_csv(fp, on_bad_lines='warn')
                return '\n'.join([' '.join(r.astype(str)) for _, r in df.iterrows()]).strip() or None
            except ImportError:
                logger.warning("pandas not installed.")
                return None
            except pd.errors.ParserError as pe:
                logger.error(f"CSV Parse Error {fp.name}: {pe}")
                return None
            except Exception as e:
                logger.error(f"CSV Error {fp.name}: {e}")
                return None

        try:
            for d_set, ext, f_type in [
                (settings.policies_dir, ext_pdf, "policy"),
                (settings.products_dir, ext_csv, "product_update")
            ]:
                if d_set and d_set.exists() and d_set.is_dir():
                    logger.info(f"Scanning {d_set}...")
                    c = 0
                    glob = "*.pdf" if f_type == "policy" else "*.csv"

                    for f in d_set.glob(glob):
                        if f.name not in processed:
                            txt = ext(f)
                            if txt:
                                initial_docs.append({
                                    'text': txt,
                                    'source': f.name,
                                    'type': f_type
                                })
                                c += 1
                            else:
                                logger.warning(f"Skipping {f_type} (extract err/empty): {f.name}")

                    if c > 0:
                        logger.info(f"Found {c} new {f_type} docs.")

            if initial_docs:
                logger.info(f"Passing {len(initial_docs)} new docs for chunking/embedding.")
                self.add_documents(initial_docs)  # Call add_documents here
            else:
                logger.info("No new initial documents found/extracted.")
        except Exception as e:
            logger.error(f"Failed loading initial docs process: {e}", exc_info=True)

    @log_execution_time
    def add_documents(self, documents: List[Dict[str, str]]):
        """Chunks documents, gets embeddings, updates index, saves chunk metadata."""
        if self.index is None or self.chunker is None or self.feature_extractor is None:
            logger.error(
                "Cannot add documents: FAISSClient essential components (index, chunker, pipeline) are not ready.")
            return

        if not documents or not isinstance(documents, list):
            logger.warning("No valid docs provided.")
            return

        chunks_to_embed = []
        chunk_meta_list = []
        total_chunks = 0
        failed_docs = 0

        logger.info(f"Chunking {len(documents)} input documents...")

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict) or 'text' not in doc or not isinstance(doc['text'], str):
                logger.warning(f"Skip invalid doc idx {i}.")
                failed_docs += 1
                continue

            original_text = doc.get('text', '').strip()
            source = doc.get('source', f'Unknown_{i + 1}')
            doc_type = doc.get('type', 'doc')

            if not original_text:
                logger.warning(f"Skip empty doc '{source}'.")
                failed_docs += 1
                continue

            doc_chunks = self.chunker.split_text(original_text)

            if not doc_chunks:
                logger.warning(f"No chunks for doc '{source}'.")
                failed_docs += 1
                continue

            logger.debug(f"Generated {len(doc_chunks)} chunks for '{source}'.")
            total_chunks += len(doc_chunks)

            for chunk_idx, chunk_txt in enumerate(doc_chunks):
                chunks_to_embed.append(chunk_txt)
                chunk_meta_list.append({
                    "chunk_text": chunk_txt,
                    "source": source,
                    "type": doc_type,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(doc_chunks)
                })

        if failed_docs > 0:
            logger.warning(f"Skipped {failed_docs} invalid/empty input documents.")

        if not chunks_to_embed:
            logger.warning("No valid text chunks generated.")
            return

        logger.info(f"Generated {total_chunks} total chunks. Embedding...")

        try:
            embeddings = self._get_embeddings(chunks_to_embed)

            if embeddings is None:
                logger.error("Embedding failed. Aborting add.")
                return

            num_added = embeddings.shape[0]
            start_idx = self.index.ntotal

            self.index.add(embeddings)
            logger.info(f"Added {num_added} chunk embeddings. New index total: {self.index.ntotal}")

            for i in range(num_added):
                chunk_meta_list[i]["index_id"] = start_idx + i

            self.documents.extend(chunk_meta_list)
            self._save_store()
        except Exception as e:
            logger.error(f"Failed adding chunks/embeddings: {e}", exc_info=True)

    @log_execution_time
    def search(self, query: str, top_k: Optional[int] = None) -> Optional[List[Dict]]:
        """Searches index for relevant chunks. Returns chunk metadata."""
        if not self.is_initialized:
            logger.error("FAISSClient not initialized. Cannot search.")
            return None

        if not query or not isinstance(query, str) or not query.strip():
            logger.warning(f"Invalid query: '{query}'")
            return None

        if self.index is None:
            logger.error("Search failed: FAISS index is None.")
            return None

        if self.index.ntotal == 0:
            logger.info("Search on empty index.")
            return []

        k_search = min(top_k if top_k is not None and top_k > 0 else self.k, self.index.ntotal)

        if k_search <= 0:
            logger.warning(f"Search k invalid/0. Cannot search.")
            return []

        logger.info(f"Searching for '{query[:100]}...' (k={k_search})")

        try:
            start = time.monotonic()
            query_emb = self._get_embeddings([query.strip()])  # Embed query

            if query_emb is None or query_emb.shape != (1, self.dimension):
                logger.error(f"Invalid query embedding.")
                return None

            logger.debug(f"Searching index ({self.index.ntotal} chunks) k={k_search}")
            distances, indices = self.index.search(query_emb, k_search)
            duration = time.monotonic() - start

            logger.info(
                f"Search completed {duration:.4f}s. Found {len(indices[0]) if indices is not None and len(indices) > 0 else 0} matches.")

            results = []

            if indices is not None and len(indices) > 0 and distances is not None and len(distances) > 0:
                for i, idx in enumerate(indices[0]):
                    if idx < 0:
                        continue

                    idx = int(idx)

                    if 0 <= idx < len(self.documents):
                        chunk_doc = self.documents[idx]
                        score = float(distances[0][i])
                        results.append({
                            'score': score,
                            'text': chunk_doc.get('chunk_text', ''),
                            'source': chunk_doc.get('source', '?'),
                            'type': chunk_doc.get('type', 'doc'),
                            'chunk_info': f"Chunk {chunk_doc.get('chunk_index', -1) + 1}/{chunk_doc.get('total_chunks', -1)}"
                        })
                    else:
                        logger.warning(f"Search index {idx} out of bounds (Max: {len(self.documents) - 1}).")

            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            return None


# --- Example Usage ---
if __name__ == "__main__":
    print("Running FAISSClient script directly for testing...")
    try:
        faiss_client = FAISSClient()  # Instantiation includes init checks now

        if faiss_client.is_initialized:  # Check the flag
            # Load initial docs is called during init if index is new
            if faiss_client.index is not None and faiss_client.index.ntotal == 0:
                print("\n--- Index is empty after init, check logs for loading/embedding issues ---")

            print("\n--- Testing search ---")
            search_query = "investment policy"
            search_results = faiss_client.search(search_query, top_k=5)

            if search_results is None:
                logger.error(f"Search failed: '{search_query}'")
            elif search_results:
                print(f"\nResults for '{search_query}' (Top {len(search_results)}):")
                for r in search_results:
                    print(
                        f"Src:{r.get('source')}, Type:{r.get('type')}, Score:{r.get('score'):.4f}, Info:{r.get('chunk_info')}\nTxt:{r.get('text', '')[:300]}...\n---")
            else:
                print(f"\nNo relevant chunks found: '{search_query}'")
        else:
            logger.error("FAISSClient did not initialize. Skipping tests.")
    except Exception as main_exc:
        logger.error(f"Error during test run: {main_exc}", exc_info=True)

    print("\nFAISSClient script test finished.")