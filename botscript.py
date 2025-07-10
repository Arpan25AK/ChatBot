import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import AutoTokenizer
from openai import OpenAI  # New OpenAI Python client from GitHub models

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Initialize OpenAI client for GitHub Models
client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=GITHUB_TOKEN,
)

# Initialize ChromaDB
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="onepiece_collection")

# Load Embedding Model (MiniLM)
model = SentenceTransformer('all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Load Text Files, Embed, Store in ChromaDB (One-Time Setup)
def load_and_store_texts(text_folder):
    for root, _, files in os.walk(text_folder):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # Token-based Chunking Logic with Overlap
                    max_chars = 1200  # Approximate to keep under tokenizer limits
                    text_pieces = []

                    for i in range(0, len(content), max_chars):
                        text_pieces.append(content[i:i + max_chars])

                    chunks = []
                    chunk_size = 400
                    chunk_overlap = 50

                    for piece in text_pieces:
                        try:
                            tokens = tokenizer.encode(piece, add_special_tokens=False, truncation=True, max_length=512)
                            for i in range(0, len(tokens), chunk_size - chunk_overlap):
                                chunk_tokens = tokens[i: i + chunk_size]
                                chunk_text = tokenizer.decode(chunk_tokens)
                                chunks.append(chunk_text)
                        except Exception as e:
                            print(f"Error tokenizing piece from {filename}: {str(e)}")
                            continue

                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            emb = model.encode(chunk, convert_to_numpy=True)
                            doc_id = f"{filename.replace('.txt', '').replace(' ', '_').lower()}_{i}"

                            collection.add(
                                documents=[chunk],
                                embeddings=[emb.tolist()],
                                metadatas=[{"source": filename}],
                                ids=[doc_id]
                            )
    print("✅ Data embedded and stored in ChromaDB.")

# Query function using GitHub Models GPT-4o via OpenAI client
def ask_bot(user_query, top_k=5):
    query_embedding = model.encode(user_query, convert_to_numpy=True)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

    context = "\n---\n".join(results["documents"][0])
    prompt = f"""You are a helpful and knowledgeable One Piece expert. Use the context provided below to answer the user's question as accurately as possible.
If the context is not sufficient, say "I'm not sure based on what I know."

Context:
{context}

Question: {user_query}
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who answers based only on the context provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: Failed to generate answer. {str(e)}"

# Run Chat Loop
if __name__ == "__main__":
    print("\n🦜 RAG Chatbot — One Piece (CTRL+C to quit)")
    text_folder = "./onepiece_data"  # Change this to your folder

    if collection.count() == 0:
        print('No documents found in the database. Creating new database...')
        load_and_store_texts(text_folder)
    else:
        print('Documents found in the database.')
        print('Count:', collection.count())

    print("Chatbot initialized. Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['exit', 'quit']:
                print("Exiting chatbot.")
                break
            answer = ask_bot(query)
            print(f"Bot: {answer}")
        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break
