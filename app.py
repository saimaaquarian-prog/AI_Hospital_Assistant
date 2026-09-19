import os
import pickle
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")


# --------------------------------------------------------------------------
# Cached loaders (run once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index_and_metadata():
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        st.error(
            f"Could not find '{index_path}' or '{meta_path}'. "
            "Make sure the faiss_index folder (built by ingest.py) is in the same "
            "directory as app.py."
        )
        st.stop()

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml (locally) or the app's Secrets "
            "settings (on Streamlit Cloud)."
        )
        st.stop()
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve_chunks(question, index, metadata, embedder, top_k=TOP_K):
    query_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        results.append(
            {
                "text": entry["text"],
                "department": entry["department"],
                "source": entry["source"],
                "page": entry["page"],
                "score": float(score),
            }
        )
    return results


def build_prompt(question, chunks):
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {c['department']} — {c['source']}, page {c['page']}]\n{c['text']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a hospital policy assistant. Answer the user's question using ONLY "
        "the information in the provided context. If the context does not contain "
        "the answer, say you don't have enough information in the policy documents "
        "to answer. Be concise and factual. When relevant, mention which department's "
        "policy the information comes from."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return system_prompt, user_prompt


def generate_answer(client, question, chunks):
    system_prompt, user_prompt = build_prompt(question, chunks)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital department policies. Answers are grounded in your uploaded PDFs.")

index, metadata = load_index_and_metadata()
embedder = load_embedder()
client = load_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['department']}** — `{s['source']}` (page {s['page']}, "
                        f"relevance {s['score']:.2f})"
                    )

# Chat input
question = st.chat_input("Ask about a hospital department policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer..."):
            chunks = retrieve_chunks(question, index, metadata, embedder)
            if not chunks:
                answer = "I couldn't find any relevant information in the policy documents."
                sources = []
            else:
                answer = generate_answer(client, question, chunks)
                sources = chunks

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"- **{s['department']}** — `{s['source']}` (page {s['page']}, "
                        f"relevance {s['score']:.2f})"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
