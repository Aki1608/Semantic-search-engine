import time
import numpy as np
import faiss
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def build_wikipedia_index():
    print("Downloading Wikipedia subset...")
    # Load the English Wikipedia dataset. 
    # We specify split="train[:50000]" to grab the first 50K articles.
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    
    # We will only embed the beginning (summary) of the text to save memory/time
    print("Extracting article summaries...")
    documents = []
    metadata = []

    # for row in dataset: # This will download the articles which will take more space.
    for row in dataset.take(50000): # Use .take(50000) to pull exactly what we need directly from the stream. No download.
        title = row['title']
        text = row['text']
        # Take the first ~500 characters as a summary for embedding
        summary = text[:500] + "..." if len(text) > 500 else text
        
        documents.append(summary)
        # Save metadata to map vectors back to human-readable info
        metadata.append({"title": title, "summary": summary, "url": row['url']})

    print("Initializing Sentence Transformer...")
    # 'all-MiniLM-L6-v2' is the gold standard for fast, high-quality sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Embedding {len(documents)} documents (This may take a few minutes)...")
    start_time = time.time()
    
    # Convert text strings into 384-dimensional dense vectors
    # batch_size and show_progress_bar help manage memory and show status
    embeddings = model.encode(documents, batch_size=256, show_progress_bar=True)
    
    # Ensure vectors are floating point numbers (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    print(f"Embedding completed in {time.time() - start_time:.2f} seconds.")

    print("Building FAISS Index...")
    # Get the dimension of the vectors (384 for MiniLM)
    dimension = embeddings.shape[1]
    
    # IndexFlatL2 performs exact nearest-neighbor search using Euclidean distance
    index = faiss.IndexFlatL2(dimension)
    
    # Load the vectors into the database
    index.add(embeddings)
    
    print("Saving index and metadata to disk...")
    # Save the database
    faiss.write_index(index, "wikipedia_50k.faiss")
    
    # Save the text metadata so we know what text belongs to which vector
    with open("wikipedia_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print("Success! Database built and ready for queries.")

if __name__ == "__main__":
    build_wikipedia_index()