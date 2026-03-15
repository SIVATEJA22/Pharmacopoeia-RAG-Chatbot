# Pharmacopoeia-RAG-Chatbot
# 🩺 Pharmacopoeia RAG Chatbot

An interactive **Streamlit** chatbot that recommends medicines from a pharmacopoeia dataset (`MEDICINES.csv`) using a **RAG (Retrieval‑Augmented Generation)** pipeline built with **Chroma**, **HuggingFace embeddings**, and **Google Gemini** via `langchain_google_genai`.[web:72][web:73]

> ⚠️ This app is for **educational purposes only** and does **not** replace professional medical advice.

---

## ✨ Features

- Chat‑style UI using `st.chat_input` and `st.chat_message` (no side panels).[web:67]
- RAG pipeline over a medicines CSV with a Chroma vector store.[web:73][web:76]
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- Gemini (`ChatGoogleGenerativeAI`) generates structured answers:
  - Recommended Medicine  
  - Strength / Power (e.g., 500 mg, extracted from `drug_content`)  
  - Doctor prescription requirement  
  - Price and manufacturer  
  - Usage & details in bullet points.[web:72]
- Shows source medicine chunks used to generate the answer.

---

## 🗂 Project Structure

```text
.
├── app.py               # Streamlit chatbot app
├── MEDICINES.csv        # Pharmacopoeia dataset
├── rag_data/
│   └── chroma_db/       # Persisted Chroma vector store
├── requirements.txt
└── .env                 # Contains GEMINI_API_KEY
```

## 🧩 How It Works

**User query**  
You type your symptoms into the chat box.

**Retrieval (RAG)**  
The app performs semantic search in Chroma using HuggingFace embeddings to fetch the top‑k relevant medicines.

**Prompting Gemini**  
Retrieved records are passed as context into a prompt template that asks Gemini to:
- Pick the best single medicine.
- Read `drug_content` to infer strength/power (e.g., 500 mg).
- Answer in a strict, line‑by‑line structured format.

**Chat UI**  
The response is rendered as a chat bubble, and an expander shows the underlying source medicines.

## 🧪 Example Questions

- fever with stomach pain  
- skin allergy and itching  
- dry cough at night  
- headache and body pain  

## ⚠️ Disclaimer

This application is **not** a medical device and is intended **only for learning and demonstration** of RAG + LLM techniques.  
Always consult a qualified healthcare professional before taking any medication.

