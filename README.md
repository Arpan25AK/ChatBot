## 🦜 One Piece Anime Chatbot using RAG (Retrieval-Augmented Generation)

An intelligent chatbot system built to answer detailed questions about the One Piece anime series using Retrieval-Augmented Generation (RAG) architecture. It leverages local text data, vector embeddings, and a powerful LLM to generate accurate, context-based responses.

### 📌 Features
* 🔍 **Retrieval-Augmented Generation**: Uses embedded local documents for context-aware question answering.
* 🧠 **LLM Integration**: Utilizes **GPT-4o** via OpenAI's GitHub-hosted model API for natural response generation.
* 📚 **ChromaDB**: Vector store for efficient similarity-based document retrieval.
* 🧩 **Custom Chunking & Embedding**: Token-aware chunking of text data with `all-mpnet-base-v2` from SentenceTransformers.

### 🚀 Demo
🦜 RAG Chatbot — One Piece (CTRL+C to quit)
Documents found in the database.
Count: 125
Chatbot initialized. Type 'exit' or 'quit' to end the conversation.

### 📁 Project Structure

├── botscript.py          # Main chatbot logic

├── onepiece_data/        # Folder with local .txt files containing One Piece content

├── chroma_db/            # Auto-generated vector database by ChromaDB

├── .env                  # Environment variables (GITHUB_TOKEN)

└── README.md             # Project documentation

### ⚙️ Setup Instructions

#### 1. Clone the Repo
git clone https://github.com/yourusername/onepiece-rag-chatbot.git
cd onepiece-rag-chatbot

#### 2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

#### 3. Install Requirements
pip install -r requirements.txt

(If no `requirements.txt`, manually install:)

pip install chromadb openai python-dotenv sentence-transformers transformers

#### 4. Add Your `.env` File
Create a `.env` file in the root directory with:
GITHUB_TOKEN=your_openai_github_api_key_here

#### 5. Add Text Files
Place `.txt` files with One Piece information in the `onepiece_data/` folder.

### ▶️ Run the Chatbot
python botscript.py

If it's the first run and no data exists in ChromaDB, it will embed and store documents automatically.

### 🧠 How It Works

1. Text data is loaded and chunked intelligently using token-aware splitting.
2. Each chunk is embedded using `all-mpnet-base-v2` and stored in ChromaDB.
3. When a user asks a question, the top-k similar chunks are retrieved.
4. GPT-4o generates a response using only the retrieved context.

### 🔐 API Model Used
* **Model**: `openai/gpt-4o`
* **Endpoint**: `https://models.github.ai/inference` (via OpenAI GitHub Client)

### 📌 Future Improvements
* Add web interface using Gradio or Streamlit
* Add automatic document preprocessing and cleaning
* Support other anime datasets or genres

### 🧑‍💻 Author
**Arpan Anand Kotian**
📧 [akarpan2005@gmail.com](mailto:akarpan2005@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/arpan-a-k-104897364/)



