"""
Embeddings Module
Handles generation of embeddings using OpenAI's API
"""
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from config.settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding model"""

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_EMBEDDING_MODEL):
        """
        Initialize the embedding generator

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model to use
        """
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file"
            )

        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
        self.model = model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        print(f"Generating embeddings for {len(texts)} texts using {self.model}...")
        embeddings = self.embeddings.embed_documents(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(query)

    def get_embeddings_object(self):
        """
        Get the underlying LangChain embeddings object

        Returns:
            OpenAIEmbeddings object
        """
        return self.embeddings


def main():
    """Test the embeddings module"""
    try:
        generator = EmbeddingGenerator()

        # Test with a sample text
        test_texts = [
            "The PM-KISAN scheme provides financial support to farmers.",
            "Eligibility criteria include owning agricultural land."
        ]

        embeddings = generator.generate_embeddings(test_texts)
        print(f"\nEmbedding dimensions: {len(embeddings[0])}")
        print(f"Sample embedding values: {embeddings[0][:5]}")

        # Test query embedding
        query = "Am I eligible for farmer schemes?"
        query_embedding = generator.generate_query_embedding(query)
        print(f"\nQuery embedding dimensions: {len(query_embedding)}")

    except ValueError as e:
        print(f"\n⚠️  {str(e)}")
        print("Please create a .env file based on .env.example and add your OpenAI API key")


if __name__ == "__main__":
    main()
