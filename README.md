# 🎥 Video Transcript RAG System

A comprehensive AI-powered system for extracting transcripts from videos and asking questions using Retrieval-Augmented Generation (RAG). Process YouTube videos or local video files, transcribe them with AI, and get intelligent answers about the content.

## 📋 Features

### 🎬 Video Processing
- **YouTube Support**: Automatically extract captions from YouTube videos
- **Local Video Upload**: Process MP4, AVI, MOV, MKV, and WebM files
- **Automatic Transcription**: Uses Whisper AI for accurate audio-to-text conversion
- **Audio Extraction**: FFmpeg integration for extracting audio from videos

### 🔍 Semantic Search
- **Vector Embeddings**: Uses sentence-transformers for semantic understanding
- **FAISS Indexing**: Fast similarity search across video content
- **Configurable Retrieval**: Adjust chunk size, similarity threshold, and result count
- **Timestamp Preservation**: All results include exact video timestamps

### 🤖 AI-Powered Answers
- **LLM Integration**: Google Gemma 2 for intelligent response generation
- **HuggingFace Models**: Leverages state-of-the-art language models
- **Context-Aware**: Answers based strictly on video content
- **Video Summarization**: Generate concise summaries of entire transcripts

### 💾 Data Management
- **Download Transcripts**: Export full transcripts as TXT files
- **Download Chunks**: Export processed chunks with timestamps
- **Session Persistence**: Maintains data during your session
- **Flexible Chunking**: Customize chunk size for your use case

## 🏗️ Architecture

```
┌─────────────────┐
│  Video Input    │
│ (YouTube/Local) │
└────────┬────────┘
         │
    ┌────▼────────────┐
    │  Transcription  │
    │  (Whisper AI)   │
    └────┬────────────┘
         │
    ┌────▼──────────┐
    │  Chunking &   │
    │  Processing   │
    └────┬──────────┘
         │
    ┌────▼──────────────────┐
    │  Embedding Generation │
    │  (Sentence Transform) │
    └────┬──────────────────┘
         │
    ┌────▼────────────┐
    │  FAISS Vector   │
    │  Store Indexing │
    └────┬────────────┘
         │
    ┌────▼──────────────┐
    │  Semantic Search │
    │  & Retrieval     │
    └────┬──────────────┘
         │
    ┌────▼───────────────┐
    │  LLM Answer Gen    │
    │  (Google Gemma 2)  │
    └────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for local video processing)
- HuggingFace API token

### Installation

```bash
# Clone the repository
git clone https://github.com/RakeshPathlavath07/video-transcript-rag.git
cd video-transcript-rag

# Install dependencies
pip install -r requirements.txt
```

### Setup HuggingFace Token

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📦 Dependencies

- **youtube-transcript-api**: Fetch captions from YouTube
- **langchain-community**: LangChain community utilities
- **langchain-huggingface**: HuggingFace integration for LangChain
- **langchain-text-splitters**: Text chunking utilities
- **faiss-cpu**: Facebook's similarity search library
- **python-dotenv**: Environment variable management
- **huggingface-hub**: HuggingFace model hub access
- **streamlit**: Web UI framework
- **transformers**: Transformer models for NLP
- **faster-whisper**: Optimized Whisper AI implementation
- **ffmpeg-python**: FFmpeg bindings for Python

## 💻 Usage

### Tab 1: 📤 Upload Video

1. Choose between **YouTube URL** or **Local Video** file
2. Paste YouTube URL or upload local video file (MP4, AVI, MOV, MKV, WebM)
3. Click **"Process"** button
4. Wait for transcription and indexing to complete
5. View processing statistics (segments, chunks, duration)

### Tab 2: ❓ Ask Questions

1. Enter your question about the video content
2. Click **"Search"** to find relevant content
3. View retrieved chunks with timestamps
4. Get AI-generated answers based on video content
5. Optional: Click **"Summarize"** for video summary

### Tab 3: 📊 View Transcript

1. View full transcript or processed chunks
2. Download transcript or chunks as TXT files
3. Each entry includes timestamps for reference

### ⚙️ Configuration (Sidebar)

- **HuggingFace Token**: Enter or load from .env
- **Chunk Size**: Number of sentences per chunk (1-10)
- **Similarity Threshold**: Minimum similarity score (0.0-1.0)
- **Number of Results (k)**: How many chunks to retrieve (1-10)

## 📁 Project Structure

```
video-transcript-rag/
├── streamlit_app.py           # Main Streamlit web application
├── general_video_main.py      # Script for processing general videos
├── youtube_video_main.py      # Script for processing YouTube videos
├── general_video.ipynb        # Jupyter notebook for general video processing
├── youtube_video.ipynb        # Jupyter notebook for YouTube video processing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 🔧 Core Components

### streamlit_app.py
The main web application providing an interactive UI for:
- Video upload and processing
- Querying video content
- Viewing and downloading transcripts

### Processing Pipeline

1. **Video Input** → YouTube URL or local file
2. **Audio Extraction** → FFmpeg converts video to WAV
3. **Transcription** → Whisper AI converts audio to text
4. **Chunking** → Splits transcript into meaningful chunks
5. **Embedding** → Sentence-transformers creates vector embeddings
6. **Indexing** → FAISS creates searchable vector index
7. **Retrieval** → Semantic search finds relevant chunks
8. **Generation** → Google Gemma 2 generates answers

## 🎯 Use Cases

- 📚 **Educational Content**: Extract key concepts from lectures
- 💼 **Business Meetings**: Summarize and search meeting recordings
- 🎤 **Podcast Analysis**: Ask questions about podcast episodes
- 📺 **Video Documentation**: Search through tutorial videos
- 🎬 **Content Analysis**: Find specific information in long videos

## 📊 Example Workflows

### YouTube Video Analysis
```bash
1. Paste: https://www.youtube.com/watch?v=VIDEO_ID
2. Click Process
3. Ask: "What are the main topics discussed?"
4. Get: AI-generated answer with timestamps
```

### Local Video Processing
```bash
1. Upload local video file
2. Configure chunk size
3. Ask: "Explain the concept of X"
4. Download transcript for reference
```

## ⚡ Performance Notes

- **First run**: Slower (models downloading)
- **Typical processing**:
  - YouTube (30 min video): 2-3 minutes
  - Local video (30 min): 3-5 minutes
  - Query response: 10-30 seconds
- **GPU recommended** for faster transcription

## 🛠️ Troubleshooting

### FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### HuggingFace token issues
- Ensure `.env` file exists with valid token
- Or provide token in Streamlit sidebar
- Get token from: https://huggingface.co/settings/tokens

### Memory issues
- Reduce chunk size
- Process shorter videos
- Use GPU for better performance

### FAISS/NumPy compatibility
```bash
pip install --upgrade faiss-cpu numpy
```

## 📝 Notes

- All processing happens locally (no cloud storage)
- Session data persists during browser session
- Supports videos up to your system's memory limits
- Timestamps are in MM:SS format

## 🔒 Privacy

- No data is sent to external servers (except HuggingFace API calls)
- Videos are processed locally
- Transcripts stored only in session memory
- No permanent data storage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rakesh Pathlavath**
- GitHub: [@RakeshPathlavath07](https://github.com/RakeshPathlavath07)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📞 Support

For issues or questions:
1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🎓 Learn More

- [LangChain Documentation](https://langchain.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace Documentation](https://huggingface.co/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Whisper AI](https://github.com/openai/whisper)

## 🚀 Future Enhancements

- [ ] Multi-video comparison
- [ ] Custom model selection
- [ ] Real-time streaming support
- [ ] API endpoint creation
- [ ] Persistent data storage
- [ ] Export to multiple formats
- [ ] Advanced filtering options
- [ ] User authentication

## 📈 Roadmap

- v1.1: Add support for more video formats
- v1.2: Implement caching for faster reprocessing
- v1.3: Add API endpoints
- v2.0: Multi-modal support (images, audio files)

---

**Made with ❤️ by Rakesh Pathlavath**
