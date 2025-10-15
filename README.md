# AI Paralegal Assistant  
*A Retrieval-Augmented Generation (RAG) Application for Legal Document Intelligence*

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-%2302569B.svg?logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![License](https://img.shields.io/badge/License-MIT-green)

---

### Overview  
**AI Paralegal Assistant** is a **Retrieval-Augmented Generation (RAG)** system that enables users to **query legal documents conversationally** — all processed **100% locally** for data privacy and zero API cost.  

It’s designed for **compliance teams, law firms, and enterprises** that handle sensitive legal data, combining **LangChain’s retrieval intelligence**, **Hugging Face embeddings**, and **Ollama (Llama 3)** for private inference.

---

### Key Features  
✅ **Private & Local AI Pipeline** — Runs entirely on your machine using Ollama and open-source embeddings.  
✅ **RAG Architecture** — Combines document retrieval + generation for accurate contextual responses.  
✅ **Streamlit Chat UI** — Upload, index, and query PDFs interactively with a clean, responsive interface.  
✅ **Automated Knowledge Ingestion** — Multi-document chunking, vectorization, and caching for instant reuse.  
✅ **Lightweight Deployment** — Fully containerizable and resource-optimized for on-prem or offline setups.  
✅ **Documentation & Logging** — Includes code-level docs, structured logs, and reproducible results.

---

### Tech Stack  
| Category | Tools Used |
|-----------|-------------|
| **Frontend / Interface** | Streamlit |
| **Backend / Logic** | Python, LangChain |
| **Model** | Ollama (Llama 3), Hugging Face Sentence Embeddings |
| **Vector Store** | ChromaDB |
| **Deployment** | Localhost / Docker (optional) |
| **Version Control** | Git + GitHub |

---

### System Architecture  
```mermaid
graph TD
A[User Uploads Legal Document] --> B[Text Chunking + Preprocessing]
B --> C[Vector Embedding via Hugging Face]
C --> D[Store in ChromaDB]
E[User Query] --> F[Retrieve Top Context Chunks]
F --> G[Llama 3 via Ollama - Context-Aware Generation]
G --> H[Streamlit UI Displays Response]
