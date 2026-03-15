import os
import json
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("🔄 Building Pharmacopoeia RAG index...")

# 1. Load dataset
df = pd.read_csv("MEDICINES.csv")
print(f"✅ Loaded {len(df)} rows from MEDICINES.csv")

# 2. Create one text chunk per row (prioritise drug_content)
def make_chunk(row):
    return (
        f"Drug Content: {row['drug_content']}\n"
        f"Disease: {row['disease_name']}\n"
        f"Medicine: {row['med_name']}\n"
        f"Price: ₹{row['price']:.2f}\n"
        f"Prescription Required: {row['prescription_required']}\n"
        f"Manufacturer: {row['drug_manufacturer']}"
    )

df["chunk"] = df.apply(make_chunk, axis=1)
chunks = df["chunk"].tolist()

metadata = df[
    ["med_name", "prescription_required", "price",
     "drug_manufacturer", "drug_content", "disease_name"]
].to_dict("records")

os.makedirs("rag_data", exist_ok=True)
with open("rag_data/drug_chunks.json", "w", encoding="utf-8") as f:
    json.dump({"chunks": chunks, "metadata": metadata}, f, ensure_ascii=False)
print("💾 Saved rag_data/drug_chunks.json")

# 3. Build Chroma index with HF embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadata,
    persist_directory="rag_data/chroma_db",
    collection_name="drugs",
)
vectorstore.persist()

print("🎉 Index built and saved to rag_data/chroma_db")
print("➡️  Next: python -m streamlit run app.py")
