"""
Retrieval Module
Handles similarity search and document retrieval
"""
from typing import List, Tuple
from langchain.schema import Document

from config.settings import TOP_K_RESULTS


class DocumentRetriever:
    """Retrieve relevant documents from vector store based on queries"""

    def __init__(self, vector_store_manager):
        """
        Initialize the document retriever

        Args:
            vector_store_manager: VectorStoreManager instance
        """
        self.vector_store_manager = vector_store_manager
        self.vector_store = vector_store_manager.get_vector_store()

        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please create or load a vector store first.")

    def retrieve_documents(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> List[Document]:
        """
        Retrieve top-k most relevant documents for a query

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of relevant Document objects
        """
        if not query.strip():
            return []

        print(f"\nSearching for top {k} relevant documents...")

        # Perform similarity search (cosine similarity via normalized vectors)
        documents = self.vector_store.similarity_search(query, k=k)

        print(f"✓ Retrieved {len(documents)} relevant documents")
        return documents

    def retrieve_documents_with_scores(
        self,
        query: str,
        k: int = TOP_K_RESULTS
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents with their similarity scores

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of tuples (Document, similarity_score)
        """
        if not query.strip():
            return []

        print(f"\nSearching for top {k} relevant documents with scores...")

        # Perform similarity search with scores
        documents_with_scores = self.vector_store.similarity_search_with_score(query, k=k)

        print(f"✓ Retrieved {len(documents_with_scores)} documents with scores")

        # Print similarity scores for debugging
        for i, (doc, score) in enumerate(documents_with_scores, 1):
            print(f"  Document {i}: Score = {score:.4f}, Source = {doc.metadata.get('source', 'unknown')}")

        return documents_with_scores

    def format_retrieved_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for LLM

        Args:
            documents: List of retrieved Document objects

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Source {i}: {source}]\n{content}")

        return "\n\n---\n\n".join(context_parts)


def main():
    """Test the retrieval module"""
    from src.embeddings import EmbeddingGenerator
    from src.retrieval.vector_store import VectorStoreManager

    try:
        # Initialize components
        embedding_gen = EmbeddingGenerator()
        vs_manager = VectorStoreManager(embedding_gen)

        # Try to load existing vector store
        if vs_manager.vector_store_exists():
            vs_manager.load_vector_store()
            retriever = DocumentRetriever(vs_manager)

            # Test query
            query = "What are the eligibility criteria for farmer schemes?"
            docs = retriever.retrieve_documents_with_scores(query)

            print("\n" + "="*60)
            print("RETRIEVED DOCUMENTS")
            print("="*60)

            for i, (doc, score) in enumerate(docs, 1):
                print(f"\nDocument {i} (Score: {score:.4f}):")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Content: {doc.page_content[:200]}...")

        else:
            print("⚠️  No vector store found. Please run the ingestion pipeline first.")

    except Exception as e:
        print(f"\n⚠️  Error: {str(e)}")


if __name__ == "__main__":
    main()
