import gradio as gr
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Global Initialization ---
print("Loading model and databases...")
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("wikipedia_50k.faiss")

with open("wikipedia_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
print("System Ready.")

def semantic_search(query):
    """Takes a user query, converts it to a vector, and searches FAISS."""
    if not query.strip():
        return "Please enter a search query."
        
    # 1. Convert user text into a mathematical vector
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    
    # 2. Search FAISS for the Top 10 closest vectors
    k = 10
    distances, indices = index.search(query_vector, k)
    
    # 3. Format the results
    results_markdown = f"### Top 10 Semantic Matches for: *\"{query}\"*\n\n"
    
    for i in range(k):
        # Get the internal ID of the matched vector
        match_id = indices[0][i]
        
        # Look up the human-readable text using that ID
        match_data = metadata[match_id]
        title = match_data['title']
        summary = match_data['summary']
        url = match_data['url']
        distance_score = distances[0][i]
        
        # Lower distance score means higher similarity
        results_markdown += f"**{i+1}. [{title}]({url})** (Distance: {distance_score:.2f})\n"
        results_markdown += f"> {summary}\n\n---\n"
        
    return results_markdown

# --- Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Wikipedia Semantic Search Engine")
    gr.Markdown(
        "This search engine uses **Sentence Transformers** and **FAISS** to search 50,000 Wikipedia articles. "
        "It doesn't look for matching keywords; it looks for matching *meaning*."
    )
    
    with gr.Row():
        with gr.Column():
            search_input = gr.Textbox(
                label="Enter a concept or question", 
                placeholder="e.g., European architecture during the Renaissance, or space travel technology"
            )
            search_btn = gr.Button("Search Semantic Meanings", variant="primary")
            
    with gr.Row():
        results_output = gr.Markdown(label="Search Results")

    # Trigger search on button click or enter key
    search_btn.click(fn=semantic_search, inputs=search_input, outputs=results_output)
    search_input.submit(fn=semantic_search, inputs=search_input, outputs=results_output)

if __name__ == "__main__":
    demo.launch()