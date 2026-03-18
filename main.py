"""
Main CLI Application for Government Scheme Eligibility Assistant
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import click
from colorama import init, Fore, Style

from src.data_ingestion import PDFIngestionPipeline
from src.embeddings import EmbeddingGenerator
from src.retrieval import VectorStoreManager, DocumentRetriever
from src.llm import SchemeEligibilityAssistant
from config.settings import RAW_DATA_DIR

# Initialize colorama for colored output
init(autoreset=True)


class SchemeAssistantCLI:
    """CLI for the Government Scheme Eligibility Assistant"""

    def __init__(self):
        """Initialize the CLI application"""
        self.embedding_generator = None
        self.vector_store_manager = None
        self.retriever = None
        self.assistant = None

    def initialize_components(self):
        """Initialize all RAG components"""
        try:
            print(f"{Fore.CYAN}Initializing components...{Style.RESET_ALL}")

            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator()
            print(f"{Fore.GREEN}✓ Embedding generator initialized{Style.RESET_ALL}")

            # Initialize vector store manager
            self.vector_store_manager = VectorStoreManager(self.embedding_generator)
            print(f"{Fore.GREEN}✓ Vector store manager initialized{Style.RESET_ALL}")

            # Initialize LLM assistant
            self.assistant = SchemeEligibilityAssistant()
            print(f"{Fore.GREEN}✓ LLM assistant initialized{Style.RESET_ALL}")

            return True

        except ValueError as e:
            print(f"{Fore.RED}✗ Initialization failed: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please make sure you have created a .env file with your OpenAI API key{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"{Fore.RED}✗ Unexpected error: {str(e)}{Style.RESET_ALL}")
            return False

    def load_or_create_vector_store(self):
        """Load existing vector store or prompt to create new one"""
        if self.vector_store_manager.vector_store_exists():
            print(f"{Fore.CYAN}Loading existing vector store...{Style.RESET_ALL}")
            self.vector_store_manager.load_vector_store()
            self.retriever = DocumentRetriever(self.vector_store_manager)
            return True
        else:
            print(f"{Fore.YELLOW}No vector store found.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please run 'python main.py ingest' first to process your documents.{Style.RESET_ALL}")
            return False

    def ingest_documents(self):
        """Ingest PDF documents and create vector store"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}DOCUMENT INGESTION PIPELINE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

        # Check if documents exist
        pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"{Fore.RED}✗ No PDF files found in {RAW_DATA_DIR}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please add PDF files containing government scheme information to:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}  {RAW_DATA_DIR}{Style.RESET_ALL}")
            return False

        print(f"{Fore.GREEN}Found {len(pdf_files)} PDF file(s) to process{Style.RESET_ALL}\n")

        # Initialize components
        if not self.initialize_components():
            return False

        # Step 1: Ingest documents
        print(f"\n{Fore.CYAN}Step 1: Loading and processing PDF documents{Style.RESET_ALL}")
        pipeline = PDFIngestionPipeline()
        documents = pipeline.ingest_documents()

        if not documents:
            print(f"{Fore.RED}✗ No documents were processed{Style.RESET_ALL}")
            return False

        # Step 2: Create vector store
        print(f"\n{Fore.CYAN}Step 2: Creating vector store with embeddings{Style.RESET_ALL}")
        self.vector_store_manager.create_vector_store(documents)

        # Step 3: Save vector store
        print(f"\n{Fore.CYAN}Step 3: Saving vector store{Style.RESET_ALL}")
        self.vector_store_manager.save_vector_store()

        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Ingestion complete! You can now use the chat interface.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")

        return True

    def chat_interface(self):
        """Interactive chat interface"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}GOVERNMENT SCHEME ELIGIBILITY ASSISTANT{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

        # Initialize components
        if not self.initialize_components():
            return

        # Load vector store
        if not self.load_or_create_vector_store():
            return

        print(f"\n{Fore.GREEN}✓ Ready to answer your questions!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Ask questions about government scheme eligibility{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'exit' or 'quit' to end the session{Style.RESET_ALL}\n")

        # Chat loop
        while True:
            try:
                # Get user query
                query = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()

                if not query:
                    continue

                if query.lower() in ['exit', 'quit', 'q']:
                    print(f"\n{Fore.GREEN}Thank you for using the Scheme Eligibility Assistant!{Style.RESET_ALL}\n")
                    break

                # Get answer
                print(f"{Fore.YELLOW}Assistant: {Style.RESET_ALL}", end="", flush=True)
                result = self.assistant.chat(query, self.retriever)

                # Display answer
                print(result["answer"])

                # Display sources
                if result["sources"]:
                    print(f"\n{Fore.BLUE}📚 Sources: {', '.join(result['sources'])}{Style.RESET_ALL}")

                print()  # Blank line for readability

            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}Thank you for using the Scheme Eligibility Assistant!{Style.RESET_ALL}\n")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")

    def query_once(self, question: str):
        """Answer a single question and exit"""
        # Initialize components
        if not self.initialize_components():
            return

        # Load vector store
        if not self.load_or_create_vector_store():
            return

        print(f"\n{Fore.CYAN}Question: {question}{Style.RESET_ALL}\n")

        # Get answer
        result = self.assistant.chat(question, self.retriever)

        # Display answer
        print(f"{Fore.YELLOW}Answer:{Style.RESET_ALL}")
        print(result["answer"])

        # Display sources
        if result["sources"]:
            print(f"\n{Fore.BLUE}📚 Sources: {', '.join(result['sources'])}{Style.RESET_ALL}\n")


@click.group()
def cli():
    """Government Scheme Eligibility Assistant CLI"""
    pass


@cli.command()
def ingest():
    """Ingest PDF documents and create vector store"""
    app = SchemeAssistantCLI()
    app.ingest_documents()


@cli.command()
def chat():
    """Start interactive chat interface"""
    app = SchemeAssistantCLI()
    app.chat_interface()


@cli.command()
@click.argument('question')
def query(question):
    """Ask a single question"""
    app = SchemeAssistantCLI()
    app.query_once(question)


@cli.command()
def info():
    """Display system information"""
    from config.settings import (
        RAW_DATA_DIR, VECTORSTORE_DIR, OPENAI_MODEL,
        OPENAI_EMBEDDING_MODEL, CHUNK_SIZE, TOP_K_RESULTS
    )

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}SYSTEM INFORMATION{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}Configuration:{Style.RESET_ALL}")
    print(f"  LLM Model: {OPENAI_MODEL}")
    print(f"  Embedding Model: {OPENAI_EMBEDDING_MODEL}")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Top K Results: {TOP_K_RESULTS}")

    print(f"\n{Fore.YELLOW}Directories:{Style.RESET_ALL}")
    print(f"  Documents: {RAW_DATA_DIR}")
    print(f"  Vector Store: {VECTORSTORE_DIR}")

    # Check for PDF files
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    print(f"\n{Fore.YELLOW}Documents Status:{Style.RESET_ALL}")
    print(f"  PDF files found: {len(pdf_files)}")

    # Check for vector store
    app = SchemeAssistantCLI()
    try:
        embedding_gen = EmbeddingGenerator()
        vs_manager = VectorStoreManager(embedding_gen)
        if vs_manager.vector_store_exists():
            print(f"  Vector store: {Fore.GREEN}✓ Exists{Style.RESET_ALL}")
        else:
            print(f"  Vector store: {Fore.RED}✗ Not created{Style.RESET_ALL}")
    except:
        print(f"  Vector store: {Fore.YELLOW}? Cannot check (API key not set){Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    cli()
