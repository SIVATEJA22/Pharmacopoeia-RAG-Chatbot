import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# ---------- Setup ----------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Pharmacopoeia RAG Chatbot", layout="wide")

# Simple CSS for assistant bubble
st.markdown(
    """
    <style>
    .bot-bubble {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        background-color: #111827;
        border: 1px solid #374151;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Pharmacopoeia RAG Chatbot")
st.caption("Ask about symptoms or health issues and get medicine suggestions from MEDICINES.csv.")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env. Please add it and restart.")
    st.stop()


@st.cache_resource
def load_rag():
    # Embeddings + Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="rag_data/chroma_db",
        embedding_function=embeddings,
        collection_name="drugs",
    )

    # Gemini LLM (use a valid name)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",      # change if you want a different model
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    # Prompt template – structured answer + strength from content
    prompt_template = """
    You are a helpful medical assistant.

    User question: {question}

    Here are relevant drug records from a pharmacopoeia-style database:
    {context}

    From this context, choose the BEST single medicine.

    The tablet strength/power (like 500 mg, 250 mg, 10 mg, 5 ml, etc.)
    is usually written inside the drug_content text. Carefully read it
    and extract the most appropriate strength/power for the recommended
    medicine. If you truly cannot find it, write "Not specified".

    Respond in this exact structured format, each field on its own line:

    Recommended Medicine: <medicine name>
    Strength / Power: <strength, e.g. 500 mg or Not specified>
    Doctor Prescription Required: <Yes/No>
    Cost of the medicine: ₹<price>
    Drug Manufactured By: <manufacturer>

    Usage & Details:
    - <symptom / use case 1>
    - <symptom / use case 2>
    - <intake instructions / precautions>

    Do not add any extra sections or headings.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"],
    )

    return vectorstore, llm, prompt


# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hi! Describe your symptoms and I’ll suggest a medicine from the pharmacopoeia.",
        }
    )

# Render previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="bot-bubble">{msg["content"].replace("\n", "<br>")}</div>',
                unsafe_allow_html=True,
            )

# ---------- User input ----------
user_query = st.chat_input("Describe your symptoms...")

if user_query:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating recommendation..."):
            try:
                vectorstore, llm, prompt = load_rag()

                # 1) Retrieve docs
                docs = vectorstore.similarity_search(user_query, k=3)

                # 2) Build context
                context_text = ""
                for d in docs:
                    meta = d.metadata
                    context_text += (
                        f"\n---\n"
                        f"Medicine: {meta.get('med_name')}\n"
                        f"Disease: {meta.get('disease_name')}\n"
                        f"Price: ₹{meta.get('price')}\n"
                        f"Prescription: {meta.get('prescription_required')}\n"
                        f"Manufacturer: {meta.get('drug_manufacturer')}\n"
                        f"Drug Content: {meta.get('drug_content')}\n"
                    )

                # 3) LLM call
                final_prompt = prompt.format(question=user_query, context=context_text)
                answer = llm.invoke(final_prompt).content.strip()

                # Show answer bubble
                st.markdown(
                    f'<div class="bot-bubble">{answer.replace("\n", "<br>")}</div>',
                    unsafe_allow_html=True,
                )

                # Optional sources
                with st.expander("Source medicines used for this answer"):
                    for i, d in enumerate(docs):
                        meta = d.metadata
                        st.markdown(
                            f"**Match {i+1}: {meta.get('med_name','N/A')} "
                            f"(₹{meta.get('price','N/A')})**"
                        )
                        st.write(meta.get("drug_content", ""))

                # Save assistant message
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                err = f"Error while running RAG: {e}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err}
                )

st.markdown("---")
st.caption("⚠️ Educational use only. Always consult a doctor.")
