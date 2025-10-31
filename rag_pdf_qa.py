
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Configuration
PDF_PATH = ""  # Replace with your PDF file
OPENAI_API_KEY = ""  # Replace with your OpenAI key
TOP_K = 3

# Set OpenAI key
openai.api_key = OPENAI_API_KEY

# Step 1: Extract text from PDF
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# Step 2: Chunk the text
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 3: Convert chunks to embeddings
def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return model, embeddings

# Step 4: Create FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index

# Step 5: Retrieve top-k chunks
def retrieve_top_k_chunks(query, model, index, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [chunks[i] for i in indices[0]]

# Step 6: Generate answer using OpenAI
def generate_answer_with_openai(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# === Main Execution ===
if __name__ == "__main__":
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Embedding chunks...")
    model, embeddings = embed_chunks(chunks)

    print("Creating FAISS index...")
    index = create_faiss_index(embeddings)

    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        #query= "give invoice number?"
        print("Retrieving relevant chunks...")
        retrieved_chunks = retrieve_top_k_chunks(query, model, index, chunks, k=TOP_K)

        print("Generating answer...")
        answer = generate_answer_with_openai(query, retrieved_chunks)
        print(f"\nAnswer: {answer}")
