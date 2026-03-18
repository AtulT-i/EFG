"""
LLM Module
Handles interaction with OpenAI's language models for answer generation
"""
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage

from config.settings import OPENAI_API_KEY, OPENAI_MODEL, TEMPERATURE


class SchemeEligibilityAssistant:
    """Assistant for answering government scheme eligibility questions"""

    def __init__(
        self,
        api_key: str = OPENAI_API_KEY,
        model: str = OPENAI_MODEL,
        temperature: float = TEMPERATURE
    ):
        """
        Initialize the LLM assistant

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Temperature for response generation (0.0 - 1.0)
        """
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file"
            )

        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=temperature
        )
        self.model = model

    def generate_answer(
        self,
        query: str,
        context_documents: List[Document]
    ) -> dict:
        """
        Generate an answer to a query using retrieved context

        Args:
            query: User's question
            context_documents: Retrieved relevant documents

        Returns:
            Dictionary containing answer and metadata
        """
        if not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "context_used": False
            }

        # Format context from retrieved documents
        context = self._format_context(context_documents)

        # Create the prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(query, context)

        # Generate response
        print(f"\nGenerating answer using {self.model}...")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm.invoke(messages)
        answer = response.content

        # Extract sources
        sources = self._extract_sources(context_documents)

        print("✓ Answer generated")

        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(context_documents) > 0,
            "num_sources": len(sources)
        }

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the assistant

        Returns:
            System prompt string
        """
        return """You are a helpful Government Scheme Eligibility Assistant. Your role is to help users understand whether they are eligible for various government schemes in India.

Guidelines:
1. Answer questions based ONLY on the provided context from official scheme documents
2. If the information is not in the context, clearly state that you don't have that information
3. Be specific about eligibility criteria, benefits, and application processes
4. Use clear, simple language that is easy to understand
5. If criteria are mentioned, list them clearly
6. Always cite the source document when providing information
7. If you're unsure, ask the user for more details about their situation

Remember: Your answers should help users understand their eligibility accurately."""

    def _create_user_prompt(self, query: str, context: str) -> str:
        """
        Create the user prompt with query and context

        Args:
            query: User's question
            context: Formatted context from retrieved documents

        Returns:
            User prompt string
        """
        if not context or context == "No relevant information found.":
            return f"""Question: {query}

Context: No relevant information found in the knowledge base.

Please inform the user that you don't have information about this scheme in your knowledge base and suggest they check official government websites or contact relevant authorities."""

        return f"""Context from Government Scheme Documents:
{context}

Question: {query}

Please provide a helpful and accurate answer based on the context above. If the context doesn't contain enough information to fully answer the question, clearly state what information is available and what is not."""

    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Document {i} - Source: {source}]\n{content}")

        return "\n\n" + ("-" * 60) + "\n\n".join([""] + context_parts)

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract unique sources from documents

        Args:
            documents: List of documents

        Returns:
            List of unique source names
        """
        sources = []
        seen = set()

        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen:
                sources.append(source)
                seen.add(source)

        return sources

    def chat(self, query: str, retriever) -> dict:
        """
        Complete RAG pipeline: retrieve context and generate answer

        Args:
            query: User's question
            retriever: DocumentRetriever instance

        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant documents
        documents = retriever.retrieve_documents(query)

        # Generate answer
        result = self.generate_answer(query, documents)

        return result


def main():
    """Test the LLM module"""
    from src.embeddings import EmbeddingGenerator
    from src.retrieval import VectorStoreManager, DocumentRetriever

    try:
        # Initialize components
        embedding_gen = EmbeddingGenerator()
        vs_manager = VectorStoreManager(embedding_gen)

        if vs_manager.vector_store_exists():
            vs_manager.load_vector_store()
            retriever = DocumentRetriever(vs_manager)
            assistant = SchemeEligibilityAssistant()

            # Test query
            query = "What is the eligibility criteria for PM-KISAN scheme?"
            result = assistant.chat(query, retriever)

            print("\n" + "="*60)
            print("ANSWER")
            print("="*60)
            print(result["answer"])
            print("\n" + "="*60)
            print(f"Sources: {', '.join(result['sources'])}")
            print("="*60)

        else:
            print("⚠️  No vector store found. Please run the ingestion pipeline first.")

    except Exception as e:
        print(f"\n⚠️  Error: {str(e)}")


if __name__ == "__main__":
    main()
