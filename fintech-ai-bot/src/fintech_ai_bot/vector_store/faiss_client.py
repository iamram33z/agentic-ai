# faiss_client.py

import faiss
import numpy as np
import json
from typing import List, Dict, Optional, Generator
from pathlib import Path
import time
import torch
from transformers import pipeline, AutoTokenizer # Import AutoTokenizer for chunking estimate
import logging
import math # For ceiling division

# --- Logger Setup ---
try:
    from fintech_ai_bot.config import settings
    from fintech_ai_bot.utils import get_logger
    logger = get_logger(__name__)
except ImportError as import_err:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import from fintech_ai_bot modules (Error: {import_err}). Using basic fallback logger and mock settings.")
    class MockSettings:
        faiss_index_path = Path("./faiss_index").resolve()
        faiss_docs_path = Path("./documents.json").resolve()
        embedding_dimension = 768
        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        hf_api_key = None
        vector_search_k = 3
        policies_dir = Path("./data/policies").resolve()
        products_dir = Path("./data/products").resolve()
        max_doc_chars_in_prompt = 5000 * 5
        log_dir = Path("./logs").resolve()
        # Add setting for chunking (optional, can hardcode)
        chunk_size_chars: int = 1500 # Approximate target size in characters
        chunk_overlap_chars: int = 200 # Overlap in characters
    settings = MockSettings()
# --- End Logger Setup ---


# --- Text Chunking Helper ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Generator[str, None, None]:
    """Yields overlapping chunks of text."""
    if not text or chunk_size <= 0:
        return
    if chunk_overlap >= chunk_size:
        logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to 0.")
        chunk_overlap = 0

    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        yield text[start:end]
        # Move start forward, accounting for overlap
        start += chunk_size - chunk_overlap
        # Break if overlap makes start point negative or static (shouldn't happen with check above)
        if start >= text_len or start < 0 or (chunk_size - chunk_overlap <= 0):
            break


class FAISSClient:
    """Client for managing FAISS index and document storage using local embeddings with mean pooling and text chunking."""

    def __init__(self):
        self.index_path: Path = settings.faiss_index_path
        self.documents_path: Path = settings.faiss_docs_path
        self.dimension: int = settings.embedding_dimension
        self.model_name: str = settings.embedding_model_name
        self.hf_api_key: Optional[str] = settings.hf_api_key
        self.k: int = settings.vector_search_k
        # Chunking settings (from config or default)
        self.chunk_size = getattr(settings, 'chunk_size_chars', 1500)
        self.chunk_overlap = getattr(settings, 'chunk_overlap_chars', 200)


        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = [] # Stores metadata for EACH CHUNK
        self.feature_extractor = None
        self.tokenizer = None # To store the tokenizer for potential future use

        if self.dimension <= 0:
             logger.critical("Embedding dimension must be positive. Cannot initialize FAISSClient.")
             raise ValueError("Embedding dimension must be positive.")

        self._initialize_embedding_pipeline()
        # Only initialize store if pipeline init was successful
        if self.feature_extractor:
            self._initialize_store()
        else:
            logger.error("Embedding pipeline failed to initialize. FAISS store cannot be initialized.")
            self.index = None
            self.documents = []


    def _initialize_embedding_pipeline(self):
        """Initializes the Hugging Face feature extraction pipeline and tokenizer."""
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing feature extraction pipeline '{self.model_name}' on device '{device}'.")
            print(f"Device set to use {device}")

            self.feature_extractor = pipeline(
                "feature-extraction",
                model=self.model_name,
                token=self.hf_api_key if self.hf_api_key else None,
                device=device
            )
            # Load tokenizer separately for potential use (like more accurate chunking)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_api_key if self.hf_api_key else None
            )

            try:
                _ = self.feature_extractor("test")
                logger.info("Feature extraction pipeline initialized and tested successfully.")
            except Exception as test_e:
                logger.warning(f"Feature extraction pipeline initialized but test run failed: {test_e}", exc_info=True)

        except Exception as e:
            logger.critical(f"Failed to initialize feature extraction pipeline or tokenizer: {e}", exc_info=True)
            self.feature_extractor = None
            self.tokenizer = None


    def _initialize_store(self):
        """Loads or creates the FAISS index and chunk document store."""
        # (Keep implementation from previous version, checks dimensions etc.)
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            if self.index_path.exists() and self.documents_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                try:
                     self.index = faiss.read_index(str(self.index_path))
                except Exception as read_err:
                     logger.critical(f"Failed to read FAISS index file {self.index_path}: {read_err}", exc_info=True)
                     self.index = None
                     self.documents = []
                     logger.warning("Treating store as non-existent due to index read error.")
                     raise # Reraise to be caught by the outer block

                if self.index is not None and self.index.d != self.dimension:
                     logger.critical(f"Loaded index dimension ({self.index.d}) does not match configured dimension ({self.dimension}). Please delete the old index files in '{self.index_path.parent}'. Aborting.")
                     raise ValueError(f"Index dimension mismatch: Loaded={self.index.d}, Config={self.dimension}. Delete old files.")

                logger.info(f"Loading existing chunk documents from {self.documents_path}")
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f) # Should contain chunk metadata
                logger.info(f"Loaded {len(self.documents)} chunk documents and index with {self.index.ntotal} vectors.")
                if self.index is not None and self.index.ntotal != len(self.documents):
                    logger.warning(f"Index vector count ({self.index.ntotal}) mismatch with chunk document count ({len(self.documents)}). Search results may be inconsistent. Consider re-indexing.")

            else:
                 if self.index is None: # Check if we need to create a new index
                     logger.info(f"Creating new FAISS index ({self.dimension} dimensions) and chunk document store.")
                     self.index = faiss.IndexFlatL2(self.dimension)
                     self.documents = [] # List to store metadata about each chunk
                     self._load_initial_documents()
                 else:
                     logger.warning("Index exists but documents file might be missing. State might be inconsistent.")

        except ValueError as ve:
             logger.critical(str(ve))
             self.index = None
             self.documents = []
             raise ve
        except Exception as e:
            logger.critical(f"FAISS store initialization failed: {e}", exc_info=True)
            logger.warning("Store initialization failed. FAISS client will have no index or documents.")
            self.index = None
            self.documents = []


    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Gets embeddings for a list of texts (presumably chunks).
        Applies MEAN POOLING. Handles 3D output. Returns None if any text fails.
        """
        # (Keep implementation from the previous version - it handles pooling and 3D)
        if self.feature_extractor is None:
            logger.error("Feature extraction pipeline is not initialized. Cannot get embeddings.")
            return None
        if not texts or not all(isinstance(t, str) for t in texts):
             logger.error(f"Invalid input to _get_embeddings: Expected list of strings, got {type(texts)}")
             return None
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
             logger.warning("Input texts list contains only empty strings.")
             return np.array([], dtype=np.float32).reshape(0, self.dimension)
        if len(valid_texts) < len(texts):
            logger.warning(f"Removed {len(texts) - len(valid_texts)} empty strings from input.")

        try:
            logger.debug(f"Generating token embeddings for {len(valid_texts)} text chunks using local pipeline.")
            token_embeddings_output = self.feature_extractor(valid_texts)

            if len(token_embeddings_output) != len(valid_texts):
                 logger.error(f"Pipeline output length ({len(token_embeddings_output)}) does not match valid input text length ({len(valid_texts)}).")
                 return None

            pooled_sentence_embeddings = []
            processing_failed = False

            for i, sentence_token_output in enumerate(token_embeddings_output):
                if processing_failed: break

                token_embeddings_array = None

                try:
                    if not isinstance(sentence_token_output, list) or not sentence_token_output:
                         logger.warning(f"Pipeline output for text chunk #{i+1} ('{valid_texts[i][:50]}...') is not a non-empty list. Skipping.")
                         processing_failed = True
                         continue

                    temp_array = np.array(sentence_token_output, dtype=np.float32)

                    if temp_array.ndim == 3 and temp_array.shape[0] == 1:
                         logger.debug(f"Detected 3D shape {temp_array.shape} for chunk #{i+1}. Squeezing batch dimension.")
                         token_embeddings_array = temp_array.squeeze(axis=0)
                    elif temp_array.ndim == 2:
                         logger.debug(f"Detected 2D shape {temp_array.shape} for chunk #{i+1}.")
                         token_embeddings_array = temp_array
                    else:
                         logger.warning(f"Token embeddings array for chunk #{i+1} ('{valid_texts[i][:50]}...') has unexpected shape {temp_array.shape} after conversion. Expected 2D or 3D with batch=1. Skipping.")
                         processing_failed = True
                         continue

                    if token_embeddings_array is None or token_embeddings_array.ndim != 2 or token_embeddings_array.shape[0] == 0:
                        logger.warning(f"Failed to obtain valid 2D token embedding array for chunk #{i+1} (Shape: {token_embeddings_array.shape if token_embeddings_array is not None else 'None'}). Skipping pooling.")
                        processing_failed = True
                        continue

                    sentence_embedding = np.mean(token_embeddings_array, axis=0)

                    if sentence_embedding.ndim == 1 and sentence_embedding.shape[0] == self.dimension:
                        pooled_sentence_embeddings.append(sentence_embedding)
                    else:
                        logger.warning(f"Pooled embedding for chunk #{i+1} ('{valid_texts[i][:50]}...') has unexpected shape {sentence_embedding.shape}. Expected ({self.dimension},). Skipping.")
                        processing_failed = True
                        continue

                except ValueError as ve:
                    logger.error(f"Could not convert token embeddings to array for chunk #{i+1} ('{valid_texts[i][:50]}...') due to inconsistent structure: {ve}. Skipping.")
                    processing_failed = True
                    continue
                except Exception as pool_exc:
                    logger.error(f"Error during processing/pooling for chunk #{i+1} ('{valid_texts[i][:50]}...'): {pool_exc}", exc_info=True)
                    processing_failed = True
                    continue
            # --- End of loop ---

            if processing_failed:
                logger.error("Aborting embedding generation as not all text chunks could be processed.")
                return None

            if not pooled_sentence_embeddings:
                logger.error("No embeddings generated despite no explicit failures detected.")
                return None

            final_embeddings_array = np.array(pooled_sentence_embeddings, dtype=np.float32)

            if final_embeddings_array.shape != (len(valid_texts), self.dimension):
                 logger.error(f"Final embedding array shape {final_embeddings_array.shape} mismatch. Expected ({len(valid_texts)}, {self.dimension}).")
                 return None

            logger.debug(f"Generated pooled embeddings of final shape: {final_embeddings_array.shape}")
            return final_embeddings_array

        except IndexError as idx_err:
             # Catch index errors specifically from the pipeline if text is too long despite chunking attempt
             logger.error(f"IndexError during pipeline processing, likely due to excessive token length in a chunk: {idx_err}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
            return None


    def _save_store(self):
        """Saves the index and chunk documents to disk."""
        # (Keep implementation from previous version)
        if self.index is None:
            logger.error("Cannot save, FAISS index is not initialized.")
            return
        if self.index.ntotal <= 0 and not self.documents: # Save empty docs file only if index is also empty
             logger.info("Skipping saving empty FAISS index and documents.")
             # Write empty list to documents file if it should be cleared
             try:
                 with open(self.documents_path, 'w', encoding='utf-8') as f: json.dump([], f)
                 logger.info(f"Wrote empty list to documents file: {self.documents_path}")
             except IOError as ioe: logger.error(f"Failed to write empty documents file {self.documents_path}: {ioe}")
             return

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving FAISS index to {self.index_path} ({self.index.ntotal} vectors)")
            faiss.write_index(self.index, str(self.index_path))

            logger.info(f"Saving {len(self.documents)} chunk documents to {self.documents_path}")
            with open(self.documents_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            logger.info("FAISS index and chunk documents saved successfully.")
        except IOError as ioe:
             logger.error(f"Failed to write FAISS index or documents file: {ioe}", exc_info=True)
        except Exception as e: # Catch other errors like Faiss runtime errors
            logger.error(f"Failed to save FAISS index or documents: {e}", exc_info=True)


    def _load_initial_documents(self):
        """Loads initial documents, extracts text, and calls add_documents (which handles chunking)."""
        # (Keep implementation from previous version - extraction logic is here)
        if self.index is None:
             logger.warning("Cannot load initial documents: FAISS index not ready.")
             return

        initial_docs_to_process = [] # Stores dicts with full text for add_documents
        processed_sources = set()

        if self.documents_path.exists() and self.documents_path.is_file():
             try:
                with open(self.documents_path, 'r', encoding='utf-8') as f:
                    existing_chunk_docs = json.load(f)
                    if isinstance(existing_chunk_docs, list):
                         # Get unique original sources from the chunk metadata
                         processed_sources.update(doc.get('source') for doc in existing_chunk_docs if isinstance(doc, dict) and 'source' in doc)
                         logger.info(f"Found {len(processed_sources)} unique sources in existing chunk documents file.")
                    else:
                         logger.warning(f"Chunk documents file {self.documents_path} does not contain a list.")
             except json.JSONDecodeError as jde:
                  logger.warning(f"Could not decode JSON from chunk documents file {self.documents_path}: {jde}")
             except Exception as e:
                  logger.warning(f"Could not read or parse existing chunk documents file {self.documents_path}: {e}")

        # --- Text Extraction Functions --- (Copied from previous version)
        def extract_text_from_pdf(file_path: Path) -> Optional[str]:
            try:
                import fitz
                if not file_path.is_file(): return None
                doc = fitz.open(file_path)
                text = "".join(page.get_text() for page in doc).strip()
                doc.close()
                if not text: logger.warning(f"No text extracted from PDF: {file_path.name}")
                logger.debug(f"Extracted text from PDF: {file_path.name} ({len(text)} chars)")
                return text if text else None
            except ImportError: logger.warning("PyMuPDF not installed. Cannot extract text from PDFs. Run 'pip install pymupdf'"); return None
            except Exception as e: logger.error(f"Error extracting text from PDF {file_path.name}: {e}"); return None

        def extract_text_from_csv(file_path: Path) -> Optional[str]:
            try:
                 import pandas as pd
                 if not file_path.is_file(): return None
                 # Add error handling for parsing specific to the error log
                 df = pd.read_csv(file_path, on_bad_lines='warn') # Warn instead of error
                 if df.empty:
                      logger.warning(f"CSV file is empty or could not be parsed cleanly: {file_path.name}")
                      return None
                 # Simpler text conversion - join rows by newline, columns by space
                 text = '\n'.join([' '.join(row.astype(str)) for _, row in df.iterrows()])
                 logger.debug(f"Extracted text from CSV: {file_path.name} ({len(text)} chars)")
                 return text.strip() if text.strip() else None
            except ImportError: logger.warning("pandas not installed. Cannot extract text from CSVs. Run 'pip install pandas'"); return None
            # Catch the specific pandas parsing error if 'warn' doesn't handle it fully
            except pd.errors.ParserError as pe:
                 logger.error(f"Pandas parsing error for CSV {file_path.name}: {pe}")
                 return None
            except Exception as e: logger.error(f"Error extracting text from CSV {file_path.name}: {e}"); return None
        # --- End Text Extraction Functions ---

        try:
            policy_dir = settings.policies_dir
            if policy_dir and policy_dir.exists() and policy_dir.is_dir():
                logger.info(f"Scanning for new policy documents in {policy_dir}...")
                count = 0
                for policy_file in policy_dir.glob("*.pdf"):
                    if policy_file.name not in processed_sources:
                        text = extract_text_from_pdf(policy_file)
                        if text:
                             # Add the dict containing the *full* text here
                             initial_docs_to_process.append({'text': text, 'source': policy_file.name, 'type': 'policy'})
                             count += 1
                        else: logger.warning(f"Skipping policy file due to extraction error or empty content: {policy_file.name}")
                if count > 0: logger.info(f"Found {count} new policy documents to process.")

            product_dir = settings.products_dir
            if product_dir and product_dir.exists() and product_dir.is_dir():
                logger.info(f"Scanning for new product documents in {product_dir}...")
                count = 0
                for product_file in product_dir.glob("*.csv"):
                    if product_file.name not in processed_sources:
                         text = extract_text_from_csv(product_file)
                         if text:
                             # Add the dict containing the *full* text here
                             initial_docs_to_process.append({'text': text, 'source': product_file.name, 'type': 'product_update'})
                             count += 1
                         else: logger.warning(f"Skipping product file due to extraction error or empty content: {product_file.name}")
                if count > 0: logger.info(f"Found {count} new product documents to process.")

            # Pass the list of documents (with full text) to add_documents
            if initial_docs_to_process:
                logger.info(f"Passing {len(initial_docs_to_process)} new initial documents to add_documents for chunking and embedding.")
                self.add_documents(initial_docs_to_process) # add_documents now handles chunking
            else:
                 logger.info("No new initial documents found or extracted successfully.")

        except Exception as e:
            logger.error(f"Failed during loading initial documents process: {e}", exc_info=True)


    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Chunks documents, gets embeddings for chunks, updates index, and saves metadata.
        Aborts if embedding fails for any chunk derived from the input documents.
        """
        if not documents: logger.info("No documents provided to add."); return
        if self.index is None: logger.error("Cannot add documents: FAISS index not initialized."); return
        if not isinstance(documents, list) or not all(isinstance(doc, dict) for doc in documents): logger.error("Invalid input: Expected list of dicts."); return
        if not all('text' in doc and isinstance(doc['text'], str) for doc in documents): logger.error("Invalid input: Docs must have 'text' field."); return
        if self.feature_extractor is None: logger.error("Cannot add documents: Embedding pipeline not initialized."); return

        chunks_to_embed = []        # List of chunk text strings
        chunk_metadata_list = []    # List of metadata dicts corresponding to chunks

        logger.info(f"Starting chunking process for {len(documents)} input documents...")
        total_chunks_generated = 0
        for i, doc in enumerate(documents):
            original_text = doc.get('text', '').strip()
            source = doc.get('source', f'Unknown_Doc_{i+1}')
            doc_type = doc.get('type', 'document')

            if not original_text:
                logger.warning(f"Skipping document '{source}' because its text is empty.")
                continue

            # Generate chunks for the current document's text
            doc_chunks = list(chunk_text(original_text, self.chunk_size, self.chunk_overlap))
            if not doc_chunks:
                 logger.warning(f"No chunks generated for document '{source}'. Text might be shorter than chunk size.")
                 # Optionally add the whole text as one chunk if it's short?
                 # if len(original_text) > 0: doc_chunks = [original_text] else: continue
                 continue # Skip if no chunks generated

            logger.debug(f"Generated {len(doc_chunks)} chunks for document '{source}'.")
            total_chunks_generated += len(doc_chunks)

            for chunk_index, chunk_text_content in enumerate(doc_chunks):
                chunks_to_embed.append(chunk_text_content)
                # Create metadata for this specific chunk
                chunk_metadata = {
                    "chunk_text": chunk_text_content, # Store the chunk text itself
                    "source": source,                 # Original source file
                    "type": doc_type,                 # Original document type
                    "chunk_index": chunk_index,       # Index of this chunk within the original doc
                    "total_chunks": len(doc_chunks)   # Total chunks for this original doc
                    # index_id will be added after inserting into FAISS
                }
                chunk_metadata_list.append(chunk_metadata)

        if not chunks_to_embed:
            logger.warning("No valid text chunks generated from the input documents.")
            return

        logger.info(f"Generated a total of {total_chunks_generated} chunks. Attempting to embed...")

        try:
            # Embed all chunks in one batch (if _get_embeddings handles batching well)
            embeddings = self._get_embeddings(chunks_to_embed)

            if embeddings is None:
                 logger.error("Failed to get embeddings for the generated chunks. Aborting add operation.")
                 # No changes made to index or self.documents yet
                 return
            # Shape validation done within _get_embeddings ensures embeddings match chunks_to_embed

            num_added = embeddings.shape[0]
            start_index = self.index.ntotal # FAISS index position for the first new vector
            self.index.add(embeddings)
            logger.info(f"Added {num_added} chunk embeddings to FAISS index. New total: {self.index.ntotal}")

            # Add index_id to metadata and extend the main documents list
            for i in range(num_added):
                # The order of embeddings matches chunk_metadata_list
                chunk_metadata_list[i]["index_id"] = start_index + i
            self.documents.extend(chunk_metadata_list) # self.documents now stores chunk metadata

            # Save the updated index and the chunk metadata
            self._save_store()

        except Exception as e:
            # Catch potential Faiss errors during add or other unexpected issues
            logger.error(f"Failed to add chunks/embeddings to FAISS index: {e}", exc_info=True)
            # Recovery is difficult here. State might be inconsistent.


    def search(self, query: str) -> Optional[List[Dict]]:
        """
        Searches the index for relevant chunks based on query embedding.
        Returns metadata of the relevant chunks.
        """
        if not query or not isinstance(query, str) or not query.strip(): logger.warning(f"Search query is invalid or empty: '{query}'"); return None
        if self.index is None: logger.error("Search failed: FAISS index is not initialized."); return None
        if self.index.ntotal == 0: logger.info("Search attempted on an empty index."); return []
        if self.feature_extractor is None: logger.error("Search failed: Embedding pipeline not initialized."); return None

        logger.info(f"Performing FAISS search for query: '{query[:100]}...'")
        try:
            start_time = time.monotonic()
            query_embedding = self._get_embeddings([query.strip()])

            if query_embedding is None or query_embedding.shape != (1, self.dimension):
                logger.error(f"Could not generate valid embedding for the search query. Shape: {query_embedding.shape if query_embedding is not None else 'None'}")
                return None

            k_search = min(self.k, self.index.ntotal)
            if k_search <= 0: logger.warning(f"Search k invalid or 0 after clamping to index size. Cannot search."); return []

            logger.debug(f"Searching index with {self.index.ntotal} vectors (chunks) for k={k_search}")
            distances, indices = self.index.search(query_embedding, k_search)
            duration = time.monotonic() - start_time
            logger.info(f"FAISS search completed in {duration:.4f}s. Found {len(indices[0]) if indices is not None and len(indices) > 0 else 0} potential matches.")

            results = []
            if indices is not None and len(indices) > 0 and distances is not None and len(distances) > 0:
                for i, idx in enumerate(indices[0]):
                    if idx < 0: continue
                    idx = int(idx)

                    if 0 <= idx < len(self.documents):
                        # Retrieve the metadata for the relevant CHUNK
                        chunk_doc = self.documents[idx]
                        if not isinstance(chunk_doc, dict): logger.warning(f"Retrieved chunk doc at index {idx} is not dict. Skipping."); continue

                        score = float(distances[0][i])
                        # Return the chunk text and its associated metadata
                        results.append({
                            'score': score,
                            'text': chunk_doc.get('chunk_text', ''), # Return the chunk text
                            'source': chunk_doc.get('source', 'Unknown'),
                            'type': chunk_doc.get('type', 'document'),
                            'chunk_info': f"Chunk {chunk_doc.get('chunk_index', -1)+1}/{chunk_doc.get('total_chunks', -1)}" # Add chunk info
                            # Consider adding index_id if needed downstream: 'index_id': idx
                        })
                    else:
                         logger.warning(f"Search returned index {idx} out of bounds for chunk documents (Max: {len(self.documents)-1}). Skipping.")

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            return None


# Example Usage (for testing script directly):
if __name__ == "__main__":
    print("Running FAISSClient script directly for testing...")

    try:
        faiss_client = FAISSClient()

        if faiss_client.index is not None:
            # Load initial docs check (if index was created anew)
            if faiss_client.index.ntotal == 0 and not faiss_client.documents:
                 print("\n--- Index is new, ensure initial documents were loaded/processed ---")
                 # _load_initial_documents was called in __init__ if index was created there

            # --- Test Search ---
            print("\n--- Testing search ---")
            search_query = "investment policy" # Example query
            search_results = faiss_client.search(search_query)

            if search_results is None:
                 logger.error(f"Search failed for query: '{search_query}'")
            elif search_results:
                print(f"\nSearch Results for: '{search_query}'")
                for result in search_results:
                    if isinstance(result, dict):
                         print(f"Source: {result.get('source', 'N/A')}, Type: {result.get('type', 'N/A')}, Score: {result.get('score', -1):.4f}")
                         print(f"Chunk Info: {result.get('chunk_info', 'N/A')}")
                         print(f"Text Preview (Chunk): {result.get('text', '')[:500]}...") # Show start of chunk
                         print("---")
                    else:
                         logger.warning(f"Malformed result: {result}")
            else:
                 print(f"\nNo relevant chunks found for: '{search_query}'")
            # --------------------
        else:
             logger.error("FAISSClient initialization failed (index is None). Skipping tests.")

    except Exception as main_exc:
         logger.error(f"Error during FAISSClient test run: {main_exc}", exc_info=True)

    print("\nFAISSClient script test finished.")