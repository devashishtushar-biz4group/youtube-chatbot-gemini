"""Streamlit UI for the Gemini-powered YouTube transcript Q&A chatbot.

Run with:  streamlit run streamlit_app.py
"""
import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Gemini models (must match what built the vector store).
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

# Each video gets its own subfolder so stores never collide.
CHROMA_BASE = "chroma_db"


def extract_video_id(url: str) -> str | None:
    """Pull the 11-character YouTube video ID out of any common URL form."""
    match = re.search(
        r"(?:v=|/embed/|youtu\.be/|/v/|/shorts/)([0-9A-Za-z_-]{11})", url
    )
    return match.group(1) if match else None


@st.cache_resource(show_spinner=False)
def build_chain_for_url(url: str):
    """Build (or reuse) a retrieval chain for a given YouTube URL.

    Cached per URL so a video is only fetched and embedded once per session.
    """
    video_id = extract_video_id(url)
    persist_dir = os.path.join(CHROMA_BASE, video_id) if video_id else None

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if persist_dir and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        # Reuse the vector store we already built for this video.
        vector_store = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings
        )
    else:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
        documents = loader.load()
        if not documents:
            raise ValueError(
                "No transcript found. The video may have captions disabled."
            )
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(documents)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions about a YouTube "
                "video. Use ONLY the following transcript context to answer. If the "
                "answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥")
st.title("🎥 YouTube Video Chatbot")
st.caption("Paste a YouTube link, load it, then ask questions about the video.")

with st.sidebar:
    st.header("1. Load a video")
    url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    load = st.button("Load video", type="primary", use_container_width=True)

    if st.session_state.get("video_url"):
        st.success(f"Loaded:\n{st.session_state['video_url']}")
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

if load:
    if not url.strip():
        st.sidebar.error("Please paste a YouTube URL first.")
    else:
        try:
            with st.spinner("Fetching transcript and building knowledge base..."):
                st.session_state["chain"] = build_chain_for_url(url.strip())
            st.session_state["video_url"] = url.strip()
            st.session_state["messages"] = []
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            st.error(f"Could not load this video: {exc}")

# Render the chat once a video is loaded.
if "chain" in st.session_state:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the video")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["chain"].invoke({"input": question})
                answer = response["answer"]
            st.markdown(answer)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("👈 Enter a YouTube URL in the sidebar and click **Load video** to begin.")
