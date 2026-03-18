# Quick Start Guide

This guide will help you get started with the Government Scheme Eligibility Assistant in just a few steps.

## Prerequisites

- Python 3.8+
- OpenAI API Key

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Add Documents

Place your government scheme PDF files in `data/raw/`:

```bash
cp /path/to/scheme-document.pdf data/raw/
```

### 4. Ingest Documents

Process the documents and create the vector store:

```bash
python main.py ingest
```

### 5. Start Asking Questions

```bash
python main.py chat
```

## Example Usage

```bash
# Interactive chat
python main.py chat

# Ask a single question
python main.py query "What is PM-KISAN?"

# Check system info
python main.py info
```

## Common Issues

### "No PDF files found"
- Add PDF files to `data/raw/` directory

### "OpenAI API key not found"
- Create `.env` file from `.env.example`
- Add your OpenAI API key

### "No vector store found"
- Run `python main.py ingest` first

## Next Steps

- Add more scheme documents to `data/raw/`
- Customize configuration in `.env`
- Check the full README.md for advanced features

## Need Help?

- Check `README.md` for detailed documentation
- Run `python main.py info` to see system status
- Open an issue on GitHub
