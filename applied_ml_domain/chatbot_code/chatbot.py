import os
import json
import numpy as np
import fitz
import streamlit as st
from PIL import Image
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# config
load_dotenv(".env")
PAPER_DIR   = "./papers"
FAISS_DIR   = "./faiss_db"
FIG_DIR     = "./figures"
FIG_INDEX   = "./figure_index.npy"
FIG_META    = "./figure_meta.json"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL  = "llama-3.3-70b-versatile"
CLIP_MODEL  = "clip-ViT-B-32"
RERANK_MODEL= "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_FIG_PX  = 100
FIG_THRESH  = 0.20

# init
_key = os.environ.get("GROQ_API_KEY")
if not _key:
    raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")
llm = ChatGroq(model=GROQ_MODEL, api_key=_key)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})

@st.cache_resource
def get_clip():
    return SentenceTransformer(CLIP_MODEL)

@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANK_MODEL)

# text pipeline
def ingest_papers():
    with st.spinner("Ingesting papers..."):
        docs = DirectoryLoader(PAPER_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
    if not docs:
        st.error("No PDFs found in papers/.")
        return None
    for d in docs:
        d.metadata["paper"] = os.path.basename(d.metadata.get("source", ""))
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(FAISS_DIR)
    st.success(f"Ingested {len(docs)} pages → {len(chunks)} chunks.")
    return store

def get_vectorstore():
    if os.path.exists(FAISS_DIR):
        try:
            return FAISS.load_local(FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True)
        except Exception:
            pass
    return ingest_papers()

# extract figures
def extract_figures():
    os.makedirs(FIG_DIR, exist_ok=True)
    figs = []
    for pdf in sorted(os.listdir(PAPER_DIR)):
        if not pdf.endswith(".pdf"):
            continue
        try:
            doc = fitz.open(os.path.join(PAPER_DIR, pdf))
            for pg, page in enumerate(doc):
                for idx, img in enumerate(page.get_images(full=True)):
                    base = doc.extract_image(img[0])
                    if base["width"] >= MIN_FIG_PX and base["height"] >= MIN_FIG_PX:
                        path = os.path.join(FIG_DIR, f"{pdf}_p{pg}_img{idx}.png")
                        with open(path, "wb") as f:
                            f.write(base["image"])
                        figs.append({"path": path, "source": pdf, "page": pg})
            doc.close()
        except Exception:
            continue
    return figs

def build_figure_index(figs, clip):
    embs, valid = [], []
    for fig in figs:
        try:
            e = clip.encode(Image.open(fig["path"]).convert("RGB"))
            embs.append(e / np.linalg.norm(e))
            valid.append(fig)
        except Exception:
            continue
    if not embs:
        return None, []
    idx = np.vstack(embs)
    np.save(FIG_INDEX, idx)
    with open(FIG_META, "w") as f:
        json.dump(valid, f)
    return idx, valid

def get_figure_index(clip):
    if os.path.exists(FIG_INDEX) and os.path.exists(FIG_META):
        try:
            idx  = np.load(FIG_INDEX)
            with open(FIG_META, "r") as f:
                meta = json.load(f)
            if idx.shape[0] > 0:
                return idx, meta
        except Exception:
            pass
    with st.spinner("Extracting and indexing figures (one-time)..."):
        figs = extract_figures()
        if not figs:
            st.warning("No figures found in PDFs.")
            return None, []
        idx, meta = build_figure_index(figs, clip)
    if idx is not None:
        st.success(f"Indexed {len(meta)} figures.")
    return idx, meta

# LLM classifiers
def _classify(query, prompt):
    try:
        return llm.invoke(prompt.format(q=query)).content.strip().lower().startswith("yes")
    except Exception:
        return False

def is_comparison_query(q):
    return _classify(q, "One word only - 'yes' if this query compares multiple papers/models/concepts, else 'no'.\nQuery: {q}\nAnswer:")

_VISUAL_KEYWORDS = {
    "figure", "diagram", "image", "chart", "plot", "graph",
    "visualization", "visualize", "show", "display", "picture", "illustration"
}

def is_visual_query(q):
    if not any(kw in q.lower() for kw in _VISUAL_KEYWORDS):
        return False
    return _classify(q, "One word only - 'yes' if the user wants to see a figure/diagram/image/chart, else 'no'.\nQuery: {q}\nAnswer:")

# retrieval + reranking
def is_clean(doc):
    t = doc.page_content
    return sum(c.isalpha() for c in t) / max(len(t), 1) > 0.4

def rerank(query, docs, reranker, top_k):
    if not docs:
        return []
    scores = reranker.predict([(query, d.page_content) for d in docs])
    return [d for _, d in sorted(zip(scores, docs), reverse=True)[:top_k]]

def retrieve(query, store, reranker):
    if is_comparison_query(query):
        broad = store.max_marginal_relevance_search(query, k=30, fetch_k=60)
        per_source = defaultdict(list)
        for d in broad:
            if is_clean(d):
                per_source[d.metadata.get("source", "")].append(d)
        chunks = [d for src_docs in per_source.values() for d in src_docs[:3]]
        if chunks:
            return rerank(query, chunks, reranker, top_k=6)
    docs = store.max_marginal_relevance_search(query, k=10, fetch_k=25)
    return rerank(query, [d for d in docs if is_clean(d)], reranker, top_k=4)

# figure retrieval
def retrieve_figures(query, docs, fig_idx, fig_meta, clip, top_k=2):
    if fig_idx is None or not fig_meta:
        return []
    top_docs = docs[:2]
    relevant_papers = {os.path.basename(d.metadata.get("source", "")) for d in top_docs}
    relevant_pages  = set()
    for d in top_docs:
        pg = d.metadata.get("page", -1)
        if pg >= 0:
            relevant_pages.update([pg - 1, pg, pg + 1])
    pool = [i for i, f in enumerate(fig_meta) if f["source"] in relevant_papers and f["page"] in relevant_pages]
    if not pool:
        pool = [i for i, f in enumerate(fig_meta) if f["source"] in relevant_papers]
    if not pool:
        pool = list(range(len(fig_meta)))
    emb    = clip.encode(query)
    emb    = emb / np.linalg.norm(emb)
    scores = (fig_idx @ emb).flatten()
    ranked = sorted(pool, key=lambda i: scores[i], reverse=True)
    return [fig_meta[i] for i in ranked[:top_k] if scores[i] >= FIG_THRESH]

# generation
def generate(query, docs):
    if not docs:
        return "I couldn't find relevant context in the papers to answer this."
    ctx = "\n\n".join(
        f'<source file="{os.path.basename(d.metadata.get("source","?"))}" '
        f'page="{d.metadata.get("page","?")}">\n{d.page_content.strip()}\n</source>'
        for d in docs
    )
    prompt = f"""You are a research assistant. Use ONLY the <source> blocks below.

RULES:
- Write a clear, plain-English answer. Do NOT copy sentences from sources verbatim.
- Do NOT output equations unless critical. End with a citation (Source: filename, Page n).
- If sources are insufficient, say "I don't have enough information."
- Relevant figures are shown below your response automatically.

<sources>
{ctx}
</sources>

Question: {query}
Answer:"""
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit UI
def main():
    st.title("RAGBOT - Research Paper Chatbot")
    st.caption("Ask questions about ML papers like Transformers, BERT, GPT, RAG, LLaMA and more")

    clip     = get_clip()
    reranker = get_reranker()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore()
    if st.session_state.vectorstore is None:
        st.error("No PDFs in papers/. Add PDFs and restart.")
        st.stop()

    if "fig_idx" not in st.session_state:
        st.session_state.fig_idx, st.session_state.fig_meta = get_figure_index(clip)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    query = st.chat_input("Ask a question about the papers...")
    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                is_visual = is_visual_query(query)
                rq = query
                if is_visual:
                    try:
                        rq = llm.invoke(
                            f"Extract only the core concept, removing words like 'show'/'display'."
                            f"\nQuery: {query}\nConcept:"
                        ).content.strip() or query
                    except Exception:
                        pass

                docs     = retrieve(rq, st.session_state.vectorstore, reranker)
                response = generate(query, docs)
                figs     = retrieve_figures(query, docs, st.session_state.fig_idx, st.session_state.fig_meta, clip) if is_visual else []

            st.write(response)

            if figs:
                st.markdown("**Relevant figures:**")
                cols = st.columns(len(figs))
                for col, fig in zip(cols, figs):
                    with col:
                        st.image(fig["path"], caption=f"{fig['source']} • Page {fig['page']}", use_container_width=True)

            with st.expander("Retrieved chunks"):
                for i, d in enumerate(docs):
                    src  = os.path.basename(d.metadata.get("source", "?"))
                    page = d.metadata.get("page", "?")
                    st.markdown(f"**{i+1}.** `{src}` · p{page}")
                    st.text(d.page_content[:300])
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response, "figures": figs})

if __name__ == "__main__":
    main()
