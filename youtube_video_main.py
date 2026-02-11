

import os
from dotenv import load_dotenv

load_dotenv()   



from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN not found. Check your .env file.")

login(token=hf_token)


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
)

chat_model = ChatHuggingFace(llm=llm)

"""## Step 1a - Indexing (Document Ingestion)"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "6S59Y0ckTm4"
api = YouTubeTranscriptApi()

try:
    transcript_list = api.fetch(video_id, languages=['en'])

    #  Timestamp-aware transcript storage
    transcript_with_timestamps = []

    for chunk in transcript_list:
        transcript_with_timestamps.append({
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.start + chunk.duration
        })

    # Optional: print sample
    for item in transcript_with_timestamps[:10]:
        print(
            f"[{item['start']:.2f}s → {item['end']:.2f}s] {item['text']}"
        )

except TranscriptsDisabled:
    print("No captions available for this video.")

except Exception as e:
    print(f"An error occurred: {e}")

"""## Step 1b - Indexing (Text Splitting)"""

def seconds_to_timestamp(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

from langchain_core.documents import Document

CHUNK_CHAR_LIMIT = 800
CHUNK_OVERLAP = 200

documents = []

current_text = ""
current_start = None

for item in transcript_with_timestamps:
    # initialize start time for a new chunk
    if current_start is None:
        current_start = item["start"]

    current_text += " " + item["text"]

    # when chunk size reached → create document
    if len(current_text) >= CHUNK_CHAR_LIMIT:
        documents.append(
            Document(
                page_content=current_text.strip(),
                metadata={
                    "start": current_start,
                    "end": item["end"]
                }
            )
        )

        # keep overlap text for next chunk
        current_text = current_text[-CHUNK_OVERLAP:]
        current_start = item["start"]

# handle remaining text
if current_text.strip():
    documents.append(
        Document(
            page_content=current_text.strip(),
            metadata={
                "start": current_start,
                "end": transcript_with_timestamps[-1]["end"]
            }
        )
    )

print(f"Total chunks created: {len(documents)}")
print("Sample chunk:")
print(documents[0])

len(documents)

documents[35]

"""## Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(documents, embeddings)

vector_store.index_to_docstore_id

vector_store.get_by_ids(['16b4f20f-392c-46a4-b9bd-3634d769aa00'])

"""## Step 2 - Retrieval"""

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

retriever

retriever.invoke('explain how many models Quantization')

"""## Step 3 - Augmentation"""

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

chat_model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
You are a video transcript analysis assistant.

Answer STRICTLY using the transcript context below.
Do NOT use outside knowledge.

TASK:
1. Decide whether the user's topic is discussed in the video.
2. If YES:
   - Clearly explain what is discussed
   - Give a short summary relevant to the question
   - List the EXACT video timestamps where it is discussed
3. If NO:
   - Say only: "NO. Sorry, the topic is not discussed in this video."

RULES:
- Start your answer with YES or NO
- Timestamps MUST be in MM:SS format
- Use ONLY timestamps present in the context

Transcript context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

question          = "is the topic about plastic is discussed ?"
retrieved_docs    = retriever.invoke(question)

retrieved_docs

context_text = "\n\n".join(
    f"[Timestamp: {seconds_to_timestamp(doc.metadata['start'])} "
    f"to {seconds_to_timestamp(doc.metadata['end'])}]\n"
    f"{doc.page_content}"
    for doc in retrieved_docs
)
context_text

final_prompt = prompt.invoke({"context": context_text, "question": question})

final_prompt

"""## Step 4 - Generation"""

final_prompt = prompt.format(
    context=context_text,
    question=question
)

answer = chat_model.invoke(final_prompt)
print(answer.content)

"""## Building a Chain"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('is topic about plastic is discussed')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | chat_model | parser

main_chain.invoke('is topic is about the mobile')

