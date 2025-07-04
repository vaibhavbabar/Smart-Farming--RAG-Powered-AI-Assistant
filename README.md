# 🌱 Smart-Farming--RAG-Powered-AI-Assistant

Empowering farmers with AI-driven solutions!  
This project is a full-stack RAG (Retrieval-Augmented Generation) application built with **Streamlit (frontend)** and **Flask + LangChain + Groq (backend)** to answer user queries from uploaded PDF documents. It also integrates a feedback loop to improve responses dynamically using model weights.

---

## 🚀 Features

- ✅ Upload and process agricultural PDFs (up to 200MB)
- 🤖 Ask questions and receive AI-powered responses using **LLaMA3-70B** via **Groq API**
- 🔎 Extracts context using **FAISS vector search**
- 📚 Displays page-level **source references**
- 🔁 Built-in **feedback system** with model improvement (RLHF)
- 🔊 Optional **Text-to-Speech** voice responses
- 📊 Live model statistics panel
- 🌐 Clean and responsive UI with **Streamlit**

---

## 🧠 Tech Stack

| Component    | Technology                                         |
|--------------|----------------------------------------------------|
| Frontend     | Streamlit, gTTS, HTML/CSS                         |
| Backend      | Flask, LangChain, FAISS, Groq (LLM), HuggingFace |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2`          |
| Vector DB    | FAISS (in-memory, can be extended to persistent) |
| Model        | `llama3-70b-8192` via Groq API                    |
| Feedback     | JSON file-based RLHF loop                         |

---

## 📁 Project Structure

```bash
.
├── frontend.py              # Streamlit UI
├── backend.py               # Flask API + LangChain logic
├── user_feedback.json       # Feedback log file (auto-generated)
├── model_weights.json       # Model weights file (auto-generated)
├── .env                     # Environment variables
└── README.md
