
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN not found in .env")

os.environ["HF_TOKEN"] = hf_token


from transformers import pipeline

llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

# Reset everything related to previous video
import os
import shutil

# Delete FAISS index folder if saved
if os.path.exists("faiss_index"):
    shutil.rmtree("faiss_index")

print(" Old video data cleared")

import subprocess
from faster_whisper import WhisperModel
import os

# ------------------ PATHS ------------------
VIDEO_PATH = "/content/Feature Construction _ Feature Splitting.mp4"
#  CHANGE THIS TO YOUR ACTUAL VIDEO PATH

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,check=True)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def build_faiss_index(chunks):
    documents = [
        Document(
            page_content=c["text"],
            metadata={"start": c["start"], "end": c["end"]}
        )
        for c in chunks
    ]
    return FAISS.from_documents(documents, embedder)

from datetime import datetime
def get_audio_path(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    name = name.replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"audio_{name}_{ts}.wav"

def process_new_video(video_path):
    audio_path = get_audio_path(video_path)

    print(" Extracting audio...")
    extract_audio(video_path, audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio extraction failed: {audio_path}")

    print(" Transcribing...")
    transcript = transcribe_audio(audio_path)

    print(" Chunking...")
    chunks = chunk_transcript(transcript)

    print(" Building FAISS index...")
    index = build_faiss_index(chunks)

    return transcript, chunks, index

from faster_whisper import WhisperModel
whisper_model = WhisperModel("medium", compute_type="int8")

def transcribe_audio(audio_path):
    segments, info = whisper_model.transcribe(audio_path, language=None)

    transcript = []
    for seg in segments:
        transcript.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end
        })

    return transcript

# ------------------ RUN PIPELINE ------------------
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

transcript, chunks, index = process_new_video(VIDEO_PATH)

print("\n===== TRANSCRIPT =====\n")
print(transcript)

def chunk_transcript(transcript, chunk_size=3):
    chunks = []
    for i in range(0, len(transcript), chunk_size):
        group = transcript[i:i+chunk_size]
        text = " ".join([g["text"] for g in group])
        start = group[0]["start"]
        end = group[-1]["end"]

        chunks.append({
            "text": text,
            "start": start,
            "end": end
        })
    return chunks

chunks = chunk_transcript(transcript)

chunks

import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [c["text"] for c in chunks]
embeddings = embedder.embed_documents(texts)


dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

def ask_question(query, index, chunks, embedder, k=3, threshold=0.7):
    query_emb = embedder.embed_query(query)
    query_emb = np.array([query_emb])  # FAISS expects 2D

    distances, indices = index.search(query_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)

        if similarity >= threshold:
            results.append({
                "text": chunks[idx]["text"],
                "start": chunks[idx]["start"],
                "end": chunks[idx]["end"],
                "similarity": similarity
            })

    return results

def generate_answer(query, retrieved_chunks):
    context = "\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""
Answer the question using ONLY the context.

Context:
{context}

Question:
{query}
"""

    return llm(prompt)[0]["generated_text"]

def summarize_video(chunks):
    full_text = " ".join([c["text"] for c in chunks])

    prompt = f"""
Summarize the following video transcript clearly:

{full_text}
"""

    return llm(prompt)[0]["generated_text"]

import os
os.environ["HF_TOKEN"] = "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"

query = "topic about feature Engineering is explained ? and summarize it ?"

retrieved = ask_question(
    query=query,
    index=index,
    chunks=chunks,
    embedder=embedder,
    k=3,
    threshold=0.5
)

if len(retrieved) == 0:
    print("Sorry, No such content is discussed")
else:
    answer = generate_answer(query, retrieved)

    print("\n Answer:")
    print(answer)

    print("\n Retrieved Text:")
    for r in retrieved:
        print(r["text"])

    print("\n Timestamps:")
    for r in retrieved:
        print(f"{r['start']:.1f}s → {r['end']:.1f}s")

