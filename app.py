import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODEL_NAME = "ibm-granite/granite-3.3-8b-instruct"  # Hugging Face Granite model


# ================= Granite LLM Client ==================
class GraniteLLMClient:
    def __init__(self, min_accept_score: float = 0.75):
        self.min_accept_score = min_accept_score
        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9
        )

    def generate_answer(self, query, context_chunks):
        if not context_chunks:
            return "I couldn’t find this information in your documents."

        good_chunks = [text for text, score in context_chunks if score >= self.min_accept_score]
        if not good_chunks:
            good_chunks = [context_chunks[0][0]]

        context = "\n\n".join(good_chunks)
        prompt = f"""
        You are an assistant that answers questions based only on the provided documents.

        Question: {query}

        Context from documents:
        {context}

        Answer concisely in 2–3 sentences. 
        If unsure, say: "Here’s the best match I found based on your documents."
        """

        try:
            response = self.pipeline(prompt)
            return response[0]["generated_text"].strip()
        except Exception as e:
            return f"Error generating answer: {e}"


# ================= Streamlit App ==================
st.set_page_config(page_title="StudyMate AI", layout="wide")

# Background
bg_image = f"""<style>.st-emotion-cache-4rsbii  {{
     background-color:#AFD2E9;
    }}
    </style> """ 
st.markdown(bg_image, unsafe_allow_html=True)

# Styling
st.markdown(
    f"""
    <style>
    .main-header {{ 
        font-size: 3rem; 
        color: #FA824C;
        text-align: center; 
        margin-bottom: 2rem; 
        font-weight: bold; 
        text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
    }}
    .upload-section {{ 
        background-color: rgba(255, 255, 255, 0.85); 
        padding: 2rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
        color: #000;
    }}
    .answer-box, .history-box {{ 
        background-color: rgba(255,255,255,0.9); 
        color: #000;  
        padding: 1.2rem; 
        border-radius: 10px; 
        border-left: 5px solid #4caf50; 
        font-size: 1.05rem; 
        line-height: 1.6; 
        margin-top: 0.5rem;
    }}
    .expander-header {{
        font-weight: 600;
        color: #1f4e79;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= Helper Functions ==================
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks, model):
    return model.encode(chunks)

def search_chunks(query, chunks, embeddings, model, top_k=3):
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in top_indices]


# ================= Main App ==================
st.markdown("<div class='main-header'>📚 AI STUDYMATE </div>", unsafe_allow_html=True)

st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ PDF uploaded and text extracted!")

    # Process doc
    chunks = chunk_text(text)
    embed_model = load_embedding_model()
    embeddings = create_embeddings(chunks, embed_model)
    llm_client = GraniteLLMClient()

    # Keep history in session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # ========== Q&A Section ==========
    st.subheader("🔎 Ask a Question about the PDF")
    query = st.text_input("Type your question here:")

    if query:
        results = search_chunks(query, chunks, embeddings, embed_model)
        answer = llm_client.generate_answer(query, results)

        # Save Q&A in history
        st.session_state.qa_history.append((query, answer))

        # Show latest answer
        st.markdown(f"<div class='answer-box'>📝 {answer}</div>", unsafe_allow_html=True)

    # Show Q&A history
    if st.session_state.qa_history:
        st.subheader("📜 Q&A History")
        for q, a in reversed(st.session_state.qa_history):
            with st.expander(f"Q: {q}"):
                st.markdown(f"<div class='history-box'><b>Answer:</b> {a}</div>", unsafe_allow_html=True)

else:
    st.info("📥 Please upload a PDF to begin.")