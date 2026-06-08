# 🎥 YouTube Video Chatbot (Gemini + LangChain + Chroma)

Ask questions about any YouTube video and get instant, transcript-grounded
answers. Paste a video link, and the app fetches its transcript, builds a local
vector database, and lets you chat with the content — either from the terminal
or through a clean web UI.

This is a **Retrieval-Augmented Generation (RAG)** application powered by
Google **Gemini** for embeddings and chat, **LangChain** for orchestration, and
**Chroma** as the local vector store.

---

## ✨ Features

- **Chat with any YouTube video** that has captions/transcripts available.
- **Two interfaces:**
  - A command-line chatbot (`app.py`).
  - A Streamlit web UI (`streamlit_app.py`) — paste a link, ask questions in a chat box.
- **Transcript-grounded answers** — responses come only from the video content; the model says "I don't know" when the answer isn't in the transcript.
- **Local & persistent vector store** — transcripts are embedded once and cached on disk (`chroma_db/`), so re-asking is instant and no re-embedding is needed.
- **Per-video isolation** in the web UI — each video gets its own vector store keyed by video ID, so you can switch between videos freely.

---

## 🧠 How It Works

```
YouTube URL
    │
    ▼
[ YoutubeLoader ]  ──►  raw transcript
    │
    ▼
[ RecursiveCharacterTextSplitter ]  ──►  overlapping text chunks (1000 chars, 200 overlap)
    │
    ▼
[ GoogleGenerativeAIEmbeddings ]  ──►  vector embeddings
    │
    ▼
[ Chroma vector store (local, persisted) ]
    │
    ▼
[ Retriever ]  ──►  top-k relevant chunks
    │
    ▼
[ create_retrieval_chain + ChatGoogleGenerativeAI ]  ──►  grounded answer
```

---

## 🛠️ Tech Stack

| Component        | Technology                                   |
| ---------------- | -------------------------------------------- |
| Language         | Python 3.14                                  |
| LLM & Embeddings | Google Gemini (`langchain-google-genai`)     |
| Orchestration    | LangChain                                    |
| Vector Store     | Chroma (`langchain-chroma`)                  |
| Transcripts      | `youtube-transcript-api` via `YoutubeLoader` |
| Web UI           | Streamlit                                    |
| Config           | `python-dotenv`                              |

**Models used:**

- Embeddings: `models/gemini-embedding-001`
- Chat: `gemini-2.5-flash`

---

## 📋 Prerequisites

- **Python 3.10+** (developed on 3.14)
- A **Google Gemini API key** — get one free at
  [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Windows users:** the
  [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
  is required by ChromaDB's native bindings. If you hit a
  `DLL load failed while importing chromadb_rust_bindings` error, install it
  (e.g. `winget install Microsoft.VCRedist.2015+.x64`) and retry.

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/devashishtushar-biz4group/youtube-chatbot-gemini.git
cd youtube-chatbot-gemini
```

### 2. Create and activate a virtual environment

```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

> ⚠️ The `.env` file is git-ignored and must **never** be committed.

### 5. (Optional) Verify your environment

```bash
python test_env.py
```

This confirms your `GOOGLE_API_KEY` is loaded correctly (the key is masked in output).

---

## 💬 Usage

### Option A — Web UI (recommended)

```bash
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501):

1. Paste a YouTube URL in the sidebar.
2. Click **Load video**.
3. Ask questions in the chat box.

### Option B — Command line

```bash
python app.py
```

The script loads the configured video (`YOUTUBE_URL` in `app.py`), builds/loads
the vector store, and starts an interactive prompt. Type your questions and
enter `exit` to quit.

---

## 📁 Project Structure

```
youtube-chatbot-gemini/
├── app.py              # CLI chatbot (interactive terminal Q&A)
├── streamlit_app.py    # Streamlit web UI
├── test_env.py         # Verifies GOOGLE_API_KEY loads from .env
├── requirements.txt    # Pinned dependencies
├── .gitignore          # Excludes .env, venv/, chroma_db/, etc.
├── .env                # Your API key (NOT committed)
└── chroma_db/          # Local vector store (auto-generated, NOT committed)
```

---

## 🔧 Configuration

You can tweak these constants at the top of `app.py` / `streamlit_app.py`:

| Constant          | Default                         | Purpose                                   |
| ----------------- | ------------------------------- | ----------------------------------------- |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001`   | Gemini embedding model                    |
| `CHAT_MODEL`      | `gemini-2.5-flash`              | Gemini chat model                         |
| `YOUTUBE_URL`     | _(sample video)_                | Default video for the CLI (`app.py`)      |
| `CHROMA_DIR`      | `chroma_db`                     | Vector store location                     |

> **Note:** The embedding model used to **query** must match the one used to
> **build** the store. If you change `EMBEDDING_MODEL`, delete `chroma_db/` so it
> rebuilds.

---

## 🩹 Troubleshooting

| Problem                                              | Fix                                                                                              |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `DLL load failed ... chromadb_rust_bindings`        | Install the Visual C++ Redistributable (see Prerequisites).                                      |
| `404 ... model is not found`                         | Your key may not have access to the configured model. Run a `ListModels` call to see what's available and update `EMBEDDING_MODEL` / `CHAT_MODEL`. |
| `RateLimitError` / quota errors                     | Check your Gemini API quota and billing.                                                         |
| "No transcript found"                               | The video has captions disabled — try a different video.                                         |
| `UnicodeEncodeError` on Windows                     | Already handled via UTF-8 reconfiguration in `app.py`.                                           |

---

## 📄 License

This project is provided as-is for educational purposes. Add a license of your
choice (e.g. MIT) if you intend to distribute it.

---

## 🙌 Acknowledgements

Built with [LangChain](https://www.langchain.com/),
[Google Gemini](https://ai.google.dev/),
[Chroma](https://www.trychroma.com/), and
[Streamlit](https://streamlit.io/).
