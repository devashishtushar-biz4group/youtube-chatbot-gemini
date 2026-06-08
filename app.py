"""YouTube transcript Q&A chatbot powered by Google Gemini + Chroma (RAG)."""
import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure non-ASCII transcript characters print on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# The YouTube video the chatbot answers questions about.
YOUTUBE_URL = "https://www.youtube.com/watch?v=LPZh9BOjkQs"

# Local directory where the Chroma vector store is persisted.
CHROMA_DIR = "chroma_db"

# Gemini models. The embedding model MUST match the one used to build the
# store, or queries won't align with the indexed vectors.
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"


def build_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Load the transcript, split it, embed it, and persist to Chroma."""
    print("Building vector store from the YouTube transcript...")

    loader = YoutubeLoader.from_youtube_url(YOUTUBE_URL, add_video_info=False)
    documents = loader.load()
    if not documents:
        raise SystemExit("No transcript found for this video.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks. Embedding and saving to disk...")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


def load_or_build_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Reuse the persisted vector store if present, otherwise build it."""
    if os.path.isdir(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Loading existing vector store from '{CHROMA_DIR}'...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return build_vector_store(embeddings)


def main() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = load_or_build_vector_store(embeddings)
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
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\nAsk questions about the video. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue

        response = qa_chain.invoke({"input": question})
        print(f"\nAnswer: {response['answer']}")


if __name__ == "__main__":
    main()
