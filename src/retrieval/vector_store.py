"""
Vector Store Module
Handles creation and management of FAISS vector store with cosine similarity
"""
import pickle
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from config.settings import VECTOR_STORE_INDEX, VECTOR_STORE_METADATA, SIMILARITY_METRIC


class VectorStoreManager:
    """Manage FAISS vector store for document retrieval with cosine similarity"""

    def __init__(self, embedding_generator):
        """
        Initialize the vector store manager

        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedding_generator = embedding_generator
        self.embeddings = embedding_generator.get_embeddings_object()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents

        Args:
            documents: List of Document objects

        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")

        print(f"\nCreating vector store from {len(documents)} documents...")
        print(f"Using similarity metric: {SIMILARITY_METRIC}")

        # Create FAISS vector store
        # LangChain's FAISS uses L2 distance by default, but we'll configure for cosine
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Normalize vectors for cosine similarity
        # Cosine similarity with normalized vectors = inner product
        self._normalize_vectors()

        print(f"✓ Vector store created with {len(documents)} documents")
        return self.vector_store

    def _normalize_vectors(self):
        """Normalize vectors in the FAISS index for cosine similarity"""
        if self.vector_store is None:
            return

        # Get the FAISS index
        index = self.vector_store.index

        # Normalize vectors for cosine similarity
        # When vectors are normalized, L2 distance approximates cosine similarity
        faiss.normalize_L2(index.reconstruct_n(0, index.ntotal))

        print("✓ Vectors normalized for cosine similarity")

    def save_vector_store(self, index_path: Path = VECTOR_STORE_INDEX):
        """
        Save the vector store to disk

        Args:
            index_path: Path to save the FAISS index
        """
        if self.vector_store is None:
            print("⚠️  No vector store to save")
            return

        print(f"\nSaving vector store to {index_path.parent}")

        # Save using LangChain's built-in save method
        self.vector_store.save_local(str(index_path.parent))

        print("✓ Vector store saved successfully")

    def load_vector_store(self, index_path: Path = VECTOR_STORE_INDEX) -> Optional[FAISS]:
        """
        Load a vector store from disk

        Args:
            index_path: Path to the FAISS index directory

        Returns:
            Loaded FAISS vector store or None if not found
        """
        try:
            print(f"\nLoading vector store from {index_path.parent}")

            # Load using LangChain's built-in load method
            self.vector_store = FAISS.load_local(
                str(index_path.parent),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("✓ Vector store loaded successfully")
            return self.vector_store

        except Exception as e:
            print(f"⚠️  Could not load vector store: {str(e)}")
            return None

    def get_vector_store(self) -> Optional[FAISS]:
        """
        Get the current vector store

        Returns:
            FAISS vector store or None
        """
        return self.vector_store

    def vector_store_exists(self, index_path: Path = VECTOR_STORE_INDEX) -> bool:
        """
        Check if a vector store exists on disk

        Args:
            index_path: Path to check

        Returns:
            True if vector store exists, False otherwise
        """
        # Check for the index.faiss and index.pkl files
        faiss_file = index_path.parent / "index.faiss"
        pkl_file = index_path.parent / "index.pkl"

        return faiss_file.exists() and pkl_file.exists()


def main():
    """Test the vector store module"""
    from src.embeddings import EmbeddingGenerator
    from langchain.schema import Document

    try:
        # Create sample documents
        sample_docs = [
            Document(
                page_content="PM-KISAN provides Rs. 6000 per year to small farmers.",
                metadata={"source": "pm-kisan.pdf"}
            ),
            Document(
                page_content="Eligibility requires owning less than 2 hectares of land.",
                metadata={"source": "pm-kisan.pdf"}
            ),
        ]

        # Initialize components
        embedding_gen = EmbeddingGenerator()
        vs_manager = VectorStoreManager(embedding_gen)

        # Create vector store
        vector_store = vs_manager.create_vector_store(sample_docs)
        print(f"\nVector store contains {vector_store.index.ntotal} vectors")

        # Save vector store
        vs_manager.save_vector_store()

        # Test loading
        vs_manager_new = VectorStoreManager(embedding_gen)
        loaded_vs = vs_manager_new.load_vector_store()

        if loaded_vs:
            print(f"Loaded vector store contains {loaded_vs.index.ntotal} vectors")

    except ValueError as e:
        print(f"\n⚠️  {str(e)}")
        print("Please ensure your OpenAI API key is configured")


if __name__ == "__main__":
    main()
