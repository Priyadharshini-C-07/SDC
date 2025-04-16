!pip install google-generativeai sentence-transformers faiss-cpu youtube-transcript-api

import os
import re
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ========== CONFIG ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyDohYPdyUX9Xhe1Ec5ssw-PRdjSapFwQzk"
genai.configure(api_key=GEMINI_API_KEY)

# ========== Get a supported model ==========
def get_supported_model():
    for model in genai.list_models():
        if (
            "generateContent" in model.supported_generation_methods and
            "vision" not in model.name.lower()  # skip vision models
        ):
            print(f"[+] Using Gemini model: {model.name}")
            return model.name
    raise Exception("❌ No valid Gemini models found that support generateContent (non-vision).")

# ========== Extract video ID ==========
def extract_video_id(url):
    patterns = [r"(?:v=|\/shorts\/|\.be\/)([a-zA-Z0-9_-]{11})"]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL: Could not extract video ID.")

# ========== Fetch transcript ==========
def get_youtube_transcript(video_id):
    print("[+] Fetching transcript...")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([item["text"] for item in transcript])

# ========== Split transcript ==========
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ========== Embed and FAISS index ==========
def build_faiss_index(chunks, embedder):
    print("[+] Embedding and indexing transcript...")
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# ========== Retrieve relevant chunks ==========
def retrieve_relevant_chunks(query, chunks, embedder, index, k=4):
    print("[+] Retrieving relevant transcript segments...")
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# ========== Summarize with Gemini ==========
def summarize_with_gemini(text, model_name):
    print("[+] Summarizing with Gemini...")
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(f"Summarize the following YouTube video content:\n\n{text}")
    return response.text

# ========== End-to-end summarizer ==========
def summarize_youtube_video(url):
    video_id = extract_video_id(url)
    transcript = get_youtube_transcript(video_id)
    chunks = split_text(transcript)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index, _ = build_faiss_index(chunks, embedder)

    relevant_chunks = retrieve_relevant_chunks("Summarize this video", chunks, embedder, index)
    combined_text = " ".join(relevant_chunks)

    model_name = get_supported_model()
    summary = summarize_with_gemini(combined_text, model_name)
    return summary

# ========== MAIN ==========
if __name__ == "__main__":
    try:
        video_url = input("Enter YouTube video URL: ").strip()
        summary = summarize_youtube_video(video_url)
        print("\n====== VIDEO SUMMARY ======\n")
        print(summary)
    except Exception as e:
        print(f"[!] Error: {e}")
