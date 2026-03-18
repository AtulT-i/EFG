"""
Document Ingestion Module
Handles loading and processing of PDF documents containing government scheme information
"""
import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class PDFIngestionPipeline:
    """Pipeline for ingesting and processing PDF documents"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the ingestion pipeline

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

        return text.strip()

    def load_pdf_documents(self, directory: Path = RAW_DATA_DIR) -> List[Document]:
        """
        Load all PDF documents from a directory

        Args:
            directory: Directory containing PDF files

        Returns:
            List of Document objects with metadata
        """
        documents = []
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            print(f"Warning: No PDF files found in {directory}")
            return documents

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")
            text = self.extract_text_from_pdf(pdf_path)

            if text:
                # Create a Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                        "type": "government_scheme"
                    }
                )
                documents.append(doc)
                print(f"  ✓ Extracted {len(text)} characters")
            else:
                print(f"  ✗ Failed to extract text")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        if not documents:
            print("No documents to split")
            return []

        print(f"\nSplitting {len(documents)} documents into chunks...")
        chunked_docs = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunked_docs)} chunks")

        return chunked_docs

    def ingest_documents(self, directory: Path = RAW_DATA_DIR) -> List[Document]:
        """
        Complete ingestion pipeline: load PDFs and split into chunks

        Args:
            directory: Directory containing PDF files

        Returns:
            List of chunked Document objects ready for embedding
        """
        print("Starting document ingestion pipeline...")
        print(f"Source directory: {directory}")

        # Load documents
        documents = self.load_pdf_documents(directory)

        if not documents:
            print("\n⚠️  No documents loaded. Please add PDF files to the data/raw directory.")
            return []

        # Split into chunks
        chunked_documents = self.split_documents(documents)

        print(f"\n✓ Ingestion complete: {len(chunked_documents)} chunks ready for embedding")
        return chunked_documents


def main():
    """Test the ingestion pipeline"""
    pipeline = PDFIngestionPipeline()
    chunks = pipeline.ingest_documents()

    if chunks:
        print("\nSample chunk:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")


if __name__ == "__main__":
    main()
