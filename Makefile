.PHONY: help setup install clean test ingest chat

# Default target
help:
	@echo "RAG Chatbot - Available Commands:"
	@echo "  make setup     - Complete setup (create venv, install deps, copy .env)"
	@echo "  make install   - Install Python dependencies"
	@echo "  make ingest    - Run document ingestion"
	@echo "  make chat      - Start the chatbot"
	@echo "  make test      - Run basic functionality tests"
	@echo "  make clean     - Clean up generated files"
	@echo "  make help      - Show this help message"

# Complete setup
setup:
	@echo "🚀 Setting up RAG Chatbot..."
	python3 -m venv .venv
	@echo "📦 Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "📄 Creating .env file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file. Please edit it with your OpenAI API key."; \
	else \
		echo "ℹ️  .env file already exists."; \
	fi
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your OpenAI API key"
	@echo "2. Run 'make ingest' to index your PDFs"
	@echo "3. Run 'make chat' to start chatting"

# Install dependencies only
install:
	@echo "📦 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run document ingestion
ingest:
	@echo "📚 Starting document ingestion..."
	python3 ingest.py
	@echo "✅ Document ingestion complete!"

# Start the chatbot
chat:
	@echo "🤖 Starting RAG Chatbot..."
	python3 chat.py

# Run basic tests
test:
	@echo "🧪 Running basic functionality tests..."
	@echo "Testing Python imports..."
	@python -c "import langchain, chromadb, openai, pypdf; print('✅ All required packages imported successfully')"
	@echo "Checking for docs directory..."
	@if [ -d "docs" ]; then \
		echo "✅ docs/ directory exists"; \
		echo "📄 PDF files found:"; \
		ls -la docs/*.pdf 2>/dev/null || echo "⚠️  No PDF files in docs/ directory"; \
	else \
		echo "❌ docs/ directory not found"; \
	fi
	@echo "Checking for .env file..."
	@if [ -f ".env" ]; then \
		echo "✅ .env file exists"; \
	else \
		echo "❌ .env file not found - copy from .env.example"; \
	fi
	@echo "Checking database..."
	@if [ -d "db" ]; then \
		echo "✅ Database directory exists"; \
	else \
		echo "ℹ️  Database not created yet - run 'make ingest' first"; \
	fi

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf db/
	rm -rf __pycache__/
	rm -rf .venv/
	rm -f *.log
	rm -f .env
	@echo "✅ Cleanup complete!"