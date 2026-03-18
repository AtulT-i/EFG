# Government Scheme Eligibility Assistant

A Retrieval-Augmented Generation (RAG) system that helps users determine their eligibility for various government schemes in India. This intelligent assistant uses OpenAI's language models combined with a vector database to provide accurate, context-aware responses based on official government scheme documents.

## 🎯 Features

- **PDF Document Processing**: Automatically processes government scheme PDFs
- **Intelligent Retrieval**: Uses FAISS with cosine similarity for accurate document retrieval
- **OpenAI Integration**: Leverages GPT models for natural language understanding and generation
- **Interactive CLI**: User-friendly command-line interface for asking questions
- **Source Citations**: Provides references to source documents for transparency
- **Flexible Configuration**: Easy customization through environment variables

## 🏗️ Architecture

This RAG system consists of the following components:

1. **Document Ingestion** (`src/data_ingestion/`): Loads and chunks PDF documents
2. **Embeddings** (`src/embeddings/`): Generates embeddings using OpenAI's embedding model
3. **Vector Store** (`src/retrieval/`): Stores and retrieves documents using FAISS with cosine similarity
4. **LLM Assistant** (`src/llm/`): Generates answers using OpenAI's language models
5. **CLI Interface** (`main.py`): Command-line interface for user interaction

## 📁 Project Structure

```
EFG/
├── config/
│   └── settings.py              # Configuration settings
├── data/
│   ├── raw/                     # Place your PDF files here
│   └── processed/               # Processed documents (auto-generated)
├── src/
│   ├── data_ingestion/          # PDF loading and processing
│   │   └── pdf_loader.py
│   ├── embeddings/              # Embedding generation
│   │   └── generator.py
│   ├── retrieval/               # Vector store and retrieval
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── llm/                     # LLM integration
│   │   └── assistant.py
│   └── utils/                   # Utility functions
├── vectorstore/                 # FAISS index (auto-generated)
├── main.py                      # Main CLI application
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AtulT-i/EFG.git
   cd EFG
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### Adding Government Scheme Documents

Place your PDF documents containing government scheme information in the `data/raw/` directory:

```bash
cp your_scheme_document.pdf data/raw/
```

## 💻 Usage

### 1. Ingest Documents

First, process your PDF documents and create the vector store:

```bash
python main.py ingest
```

This command will:
- Load all PDF files from `data/raw/`
- Split documents into chunks
- Generate embeddings using OpenAI
- Create a FAISS vector store with cosine similarity
- Save the vector store for future use

### 2. Interactive Chat

Start an interactive chat session:

```bash
python main.py chat
```

Example conversation:
```
You: What is the PM-KISAN scheme?
Assistant: PM-KISAN is a scheme that provides financial support...

You: Am I eligible for PM-KISAN if I own 1.5 hectares of land?
Assistant: Based on the scheme documents, farmers owning...

You: exit
```

### 3. Single Query

Ask a single question:

```bash
python main.py query "What are the eligibility criteria for the Ayushman Bharat scheme?"
```

### 4. System Information

Check system status and configuration:

```bash
python main.py info
```

## ⚙️ Configuration

Edit `.env` to customize the system:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo              # Or gpt-4, gpt-4-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Document Processing
CHUNK_SIZE=1000                         # Characters per chunk
CHUNK_OVERLAP=200                       # Overlap between chunks

# Retrieval
TOP_K_RESULTS=3                         # Number of relevant chunks to retrieve

# LLM Response
TEMPERATURE=0.3                         # 0.0 (deterministic) to 1.0 (creative)
```

## 🔍 How It Works

### RAG Pipeline

1. **User Query** → User asks a question about scheme eligibility
2. **Query Embedding** → Query is converted to an embedding vector
3. **Similarity Search** → FAISS finds the most relevant document chunks using cosine similarity
4. **Context Retrieval** → Top-K relevant chunks are retrieved
5. **Answer Generation** → OpenAI generates an answer using the retrieved context
6. **Response** → User receives an answer with source citations

### Cosine Similarity

The system uses **cosine similarity** to find relevant documents:
- Vectors are normalized in the FAISS index
- Cosine similarity measures the angle between query and document vectors
- More similar documents have higher cosine scores (closer to 1.0)

## 📚 Example Queries

Here are some example questions you can ask:

- "What is the PM-KISAN scheme?"
- "Am I eligible for Ayushman Bharat if my income is below 5 lakhs?"
- "What documents do I need to apply for the Pradhan Mantri Awas Yojana?"
- "What are the benefits of the PM-KISAN scheme?"
- "How do I apply for the Atal Pension Yojana?"

## 🛠️ Development

### Running Individual Modules

Test individual components:

```bash
# Test document ingestion
python -m src.data_ingestion.pdf_loader

# Test embeddings
python -m src.embeddings.generator

# Test vector store
python -m src.retrieval.vector_store

# Test retriever
python -m src.retrieval.retriever

# Test LLM assistant
python -m src.llm.assistant
```

### Adding New Features

The modular architecture makes it easy to extend:

- **New document types**: Modify `src/data_ingestion/pdf_loader.py`
- **Different embeddings**: Update `src/embeddings/generator.py`
- **Custom retrieval**: Modify `src/retrieval/retriever.py`
- **LLM customization**: Update `src/llm/assistant.py`

## 🔒 Security Notes

- **API Keys**: Never commit your `.env` file to version control
- **Data Privacy**: Ensure you have rights to process the PDF documents
- **OpenAI Usage**: Be aware of OpenAI's pricing and usage policies

## 📝 Phase 1 vs Phase 2

### Phase 1 (Current Implementation) ✅
- PDF document support only
- CLI interface
- OpenAI integration
- FAISS with cosine similarity
- Local deployment

### Phase 2 (Future Enhancements) 🔮
- Support for multiple document formats (DOCX, TXT, HTML)
- Web UI (Streamlit/FastAPI)
- Advanced features (conversation history, user feedback)
- Deployment configurations (Docker, cloud)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- OpenAI for GPT and embedding models
- LangChain for RAG framework
- FAISS for efficient similarity search

## 📧 Support

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ to help people access government schemes they're eligible for**
