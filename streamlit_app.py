import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Video Transcript RAG",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎥 Video Transcript RAG System")
st.markdown("Extract, process, and ask questions about video content using AI-powered RAG")

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # HuggingFace Token
    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Get your token from https://huggingface.co/settings/tokens"
    )
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    st.divider()
    
    # Processing parameters
    st.subheader("Processing Parameters")
    chunk_size = st.slider(
        "Chunk Size (sentences)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of sentences to combine per chunk"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum similarity score for retrieved chunks"
    )
    
    k_results = st.slider(
        "Number of Results (k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of chunks to retrieve for each query"
    )
    
    st.divider()
    st.caption("💡 For local videos, ensure FFmpeg is installed")

# ============================================
# INITIALIZE SESSION STATE
# ============================================
if "video_data" not in st.session_state:
    st.session_state.video_data = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

# ============================================
# MAIN TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["📤 Upload Video", "❓ Ask Questions", "📊 View Transcript"])

# ============================================
# TAB 1: UPLOAD VIDEO
# ============================================
with tab1:
    st.header("Upload & Process Video")
    
    video_source = st.radio(
        "Choose video source:",
        ["YouTube URL", "Upload Local Video"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if video_source == "YouTube URL":
            video_input = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste a YouTube video URL"
            )
        else:
            video_input = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Supported formats: MP4, AVI, MOV, MKV, WebM"
            )
    
    with col2:
        if st.button("▶️ Process", use_container_width=True, type="primary"):
            if not hf_token:
                st.error("❌ Please provide HuggingFace Token in sidebar")
            elif not video_input:
                st.error("❌ Please provide a video source")
            else:
                with st.spinner("🔄 Processing video..."):
                    try:
                        if video_source == "YouTube URL":
                            st.info("📝 Processing YouTube video...")
                            from youtube_transcript_api import YouTubeTranscriptApi
                            
                            # Extract video ID
                            video_id = video_input.split("v=")[-1].split("&")[0]
                            
                            # Fetch transcript
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                            
                            st.session_state.transcript = []
                            for item in transcript_list:
                                st.session_state.transcript.append({
                                    "text": item["text"],
                                    "start": item.get("start", 0),
                                    "end": item.get("start", 0) + item.get("duration", 0)
                                })
                        else:
                            st.info("🎬 Processing local video...")
                            # Save uploaded file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                                tmp.write(video_input.read())
                                video_path = tmp.name
                            
                            # Extract audio and transcribe
                            from faster_whisper import WhisperModel
                            import subprocess
                            
                            audio_path = video_path.replace(".mp4", ".wav")
                            
                            # Extract audio using FFmpeg
                            st.write("Extracting audio...")
                            subprocess.run([
                                "ffmpeg", "-y", "-i", video_path,
                                "-ar", "16000", "-ac", "1", audio_path
                            ], capture_output=True)
                            
                            # Transcribe
                            st.write("Transcribing audio...")
                            model = WhisperModel("medium", compute_type="int8")
                            segments, info = model.transcribe(audio_path)
                            
                            st.session_state.transcript = []
                            for segment in segments:
                                st.session_state.transcript.append({
                                    "text": segment.text.strip(),
                                    "start": segment.start,
                                    "end": segment.end
                                })
                            
                            # Cleanup
                            os.remove(video_path)
                            os.remove(audio_path)
                        
                        # Process transcript into chunks
                        st.write("Creating chunks...")
                        st.session_state.chunks = []
                        for i in range(0, len(st.session_state.transcript), chunk_size):
                            group = st.session_state.transcript[i:i+chunk_size]
                            text = " ".join([g["text"] for g in group])
                            st.session_state.chunks.append({
                                "text": text,
                                "start": group[0]["start"],
                                "end": group[-1]["end"]
                            })
                        
                        # Build vector store
                        st.write("Building vector store...")
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        from langchain_community.vectorstores import FAISS
                        from langchain_core.documents import Document
                        
                        st.session_state.embedder = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        
                        documents = [
                            Document(
                                page_content=c["text"],
                                metadata={"start": c["start"], "end": c["end"]}
                            )
                            for c in st.session_state.chunks
                        ]
                        
                        st.session_state.vector_store = FAISS.from_documents(
                            documents,
                            st.session_state.embedder
                        )
                        
                        st.success("✅ Video processed successfully!")
                        
                        # Display stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Segments", len(st.session_state.transcript))
                        with col2:
                            st.metric("Chunks Created", len(st.session_state.chunks))
                        with col3:
                            total_duration = sum([s["end"] - s["start"] for s in st.session_state.transcript])
                            st.metric("Duration", f"{int(total_duration//60)}m {int(total_duration%60)}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")

# ============================================
# TAB 2: ASK QUESTIONS
# ============================================
with tab2:
    st.header("Ask Questions About Video")
    
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please process a video first in the 'Upload Video' tab")
    else:
        query = st.text_input(
            "Ask a question about the video:",
            placeholder="e.g., What is machine learning?",
            help="Ask any question about the video content"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔍 Search", use_container_width=True, type="primary"):
                if query:
                    with st.spinner("Searching..."):
                        try:
                            # Retrieve relevant chunks
                            retriever = st.session_state.vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": k_results}
                            )
                            
                            retrieved_docs = retriever.invoke(query)
                            
                            # Filter by threshold
                            relevant_chunks = []
                            for doc in retrieved_docs:
                                relevant_chunks.append({
                                    "text": doc.page_content,
                                    "start": doc.metadata.get("start", 0),
                                    "end": doc.metadata.get("end", 0)
                                })
                            
                            if not relevant_chunks:
                                st.info("ℹ️ No relevant content found for this query")
                            else:
                                st.success(f"✅ Found {len(relevant_chunks)} relevant chunks")
                                
                                # Display results
                                for i, chunk in enumerate(relevant_chunks, 1):
                                    with st.expander(f"📍 Result {i} [{chunk['start']:.0f}s - {chunk['end']:.0f}s]"):
                                        st.write(chunk["text"])
                                        st.caption(f"⏱️ {int(chunk['start']//60):02d}:{int(chunk['start']%60):02d} - {int(chunk['end']//60):02d}:{int(chunk['end']%60):02d}")
                                
                                # Generate answer using LLM
                                try:
                                    st.divider()
                                    st.subheader("🤖 AI Generated Answer")
                                    
                                    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
                                    from langchain_core.prompts import PromptTemplate
                                    
                                    llm = HuggingFaceEndpoint(
                                        repo_id="google/gemma-2-2b-it",
                                        task="text-generation",
                                        max_new_tokens=512,
                                        temperature=0.7,
                                    )
                                    
                                    context = "\n".join([c["text"] for c in relevant_chunks])
                                    
                                    prompt_template = f"""Answer STRICTLY using ONLY the provided context.

Context:
{context}

Question: {query}

Answer:"""
                                    
                                    response = llm(prompt_template)
                                    st.write(response)
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Could not generate AI answer: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("📋 Summarize", use_container_width=True):
                with st.spinner("Generating summary..."):
                    try:
                        from langchain_huggingface import HuggingFaceEndpoint
                        
                        llm = HuggingFaceEndpoint(
                            repo_id="google/gemma-2-2b-it",
                            task="text-generation",
                            max_new_tokens=512,
                            temperature=0.7,
                        )
                        
                        full_text = " ".join([c["text"] for c in st.session_state.chunks])
                        
                        prompt = f"""Summarize this video transcript concisely:

{full_text}

Summary:"""
                        
                        summary = llm(prompt)
                        st.subheader("📝 Video Summary")
                        st.write(summary)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")

# ============================================
# TAB 3: VIEW TRANSCRIPT
# ============================================
with tab3:
    st.header("Transcript & Chunks")
    
    if st.session_state.transcript is None:
        st.warning("⚠️ Please process a video first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Transcript (TXT)", use_container_width=True):
                transcript_text = "\n".join([f"[{int(s['start']//60):02d}:{int(s['start']%60):02d}] {s['text']}" for s in st.session_state.transcript])
                st.download_button(
                    label="Download Full Transcript",
                    data=transcript_text,
                    file_name="transcript.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📥 Download Chunks (TXT)", use_container_width=True):
                chunks_text = "\n\n---\n\n".join([f"[{int(c['start']//60):02d}:{int(c['start']%60):02d} - {int(c['end']//60):02d}:{int(c['end']%60):02d}]\n{c['text']}" for c in st.session_state.chunks])
                st.download_button(
                    label="Download Chunks",
                    data=chunks_text,
                    file_name="chunks.txt",
                    mime="text/plain"
                )
        
        st.divider()
        
        view_type = st.radio("View:", ["Full Transcript", "Chunks"], horizontal=True)
        
        if view_type == "Full Transcript":
            st.subheader("📝 Full Transcript")
            for segment in st.session_state.transcript:
                st.write(f"**[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]** {segment['text']}")
        else:
            st.subheader("📦 Chunks")
            for i, chunk in enumerate(st.session_state.chunks, 1):
                with st.expander(f"Chunk {i} [{int(chunk['start']//60):02d}:{int(chunk['start']%60):02d} - {int(chunk['end']//60):02d}:{int(chunk['end']%60):02d}]"):
                    st.write(chunk["text"])

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("🔗 Video Transcript RAG | Powered by LangChain, HuggingFace, and Streamlit")
