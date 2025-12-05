"""
Medical Drug Information RAG Chat - Streamlit Version
Minimal code implementation
"""

import os
import streamlit as st
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Medical Drug Information",
    page_icon="💊",
    layout="centered"
)

# Configuration
CMU_BASE_URL = 'https://ai-gateway.andrew.cmu.edu/'

# Corpus path - adjust if needed
CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus"

# Load API Key
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.environ.get('OPENAI_API_KEY', '')

# Initialize RAG System
@st.cache_resource
def initialize_rag():
    """Initialize RAG system (cached)"""
    if not API_KEY:
        st.error("❌ OPENAI_API_KEY not found. Set it as environment variable or in .env file")
        st.stop()
    
    with st.spinner("🔄 Initializing RAG System..."):
        # Load documents
        corpus_path = CORPUS_PATH
        
        # Check if path exists
        if not corpus_path.exists():
            st.error(f"❌ Corpus path not found: {corpus_path.absolute()}")
            st.info(f"Please ensure 'data/corpus' directory exists")
            st.stop()
        
        # Load all .txt files
        docs = []
        txt_files = list(corpus_path.glob("*.txt"))
        
        if not txt_files:
            st.error(f"❌ No .txt files found in {corpus_path.absolute()}")
            st.info("Please add medical drug information documents (.txt files) to the corpus directory")
            st.stop()
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": file_path.name}
                        ))
            except Exception as e:
                st.warning(f"⚠️ Could not read {file_path.name}: {e}")
        
        if not docs:
            st.error("❌ No valid documents loaded")
            st.stop()
        
        # st.success(f"✓ Loaded {len(docs)} documents")
        
        # Split and index
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        if not splits:
            st.error("❌ No document chunks created")
            st.stop()
        
        # st.success(f"✓ Created {len(splits)} chunks")
        
        try:
            embeddings = OpenAIEmbeddings(
                model="azure/text-embedding-3-small",
                api_key=API_KEY,
                base_url=CMU_BASE_URL
            )
            vectorstore = FAISS.from_documents(splits, embeddings)
            # st.success("✓ Vector store ready")
        except Exception as e:
            st.error(f"❌ Error creating vector store: {e}")
            st.stop()
        
        # Initialize LLM
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini-2024-07-18",
                api_key=API_KEY,
                base_url=CMU_BASE_URL,
                temperature=0.1,
                max_tokens=500
            )
            # st.success("✓ LLM ready")
        except Exception as e:
            st.error(f"❌ Error initializing LLM: {e}")
            st.stop()
        
        return vectorstore, llm

vectorstore, llm = initialize_rag()

# UI
st.title("💊 Medical Drug Information System")
st.caption("Powered by RAG • FDA-Approved Information & Interactions")

# Warning
st.warning("⚠️ **Medical Disclaimer:** This information is for educational purposes only. Always consult healthcare professionals.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I'm your Medical Drug Information Assistant. I can help you with medication dosages, side effects, drug interactions, contraindications, and usage guidelines. What would you like to know?"
    }]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick examples
if len(st.session_state.messages) == 1:
    st.subheader("Quick questions:")
    cols = st.columns(2)
    examples = [
        "What is the typical clopidogrel loading dose used after PCI?",
        "Which monitoring parameter helps guide titration of basal insulin glargine?",
        "What daily timing is recommended for taking simvastatin?",
        "Which PPI is preferred over omeprazole when a patient is taking clopidogrel?"
    ]
    for idx, example in enumerate(examples):
        if cols[idx % 2].button(example, key=f"ex_{idx}"):
            st.session_state.user_input = example
            st.rerun()

# Chat input
if prompt := st.chat_input("Ask about drug dosages, interactions, side effects..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching medical database..."):
            try:
                # Retrieve docs
                results = vectorstore.similarity_search(prompt, k=3)
                
                if results:
                    # Build context
                    context_parts = []
                    sources = []
                    for result in results:
                        source = result.metadata.get("source", "unknown")
                        if source not in sources:
                            sources.append(source)
                        context_parts.append(f"[{source}]\n{result.page_content}")
                    
                    context = "\n\n---\n\n".join(context_parts)
                    
                    # Generate answer
                    prompt_text = f"""You are a medical information assistant. Answer based on the provided documentation.

DOCUMENTATION:
{context}

QUESTION: {prompt}

Provide a clear, accurate answer based only on the documentation above.

ANSWER:"""
                    
                    response = llm.invoke(prompt_text)
                    answer = response.content.strip()
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        st.divider()
                        st.caption("📚 **Sources:** " + ", ".join(sources))
                    
                    full_response = answer + (f"\n\n📚 **Sources:** " + ", ".join(sources) if sources else "")
                else:
                    response = llm.invoke(f"QUESTION: {prompt}\n\nANSWER:")
                    answer = response.content.strip()
                    st.markdown(answer)
                    full_response = answer
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This system uses Retrieval-Augmented Generation (RAG) to provide accurate 
    medical drug information from FDA-approved sources.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your Medical Drug Information Assistant. What would you like to know?"
        }]
        st.rerun()
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"**Script Location:** {Path(__file__).resolve()}")
        st.write(f"**Corpus Path (Config):** {CORPUS_PATH.absolute()}")
        st.write(f"**Path Exists:** {CORPUS_PATH.exists()}")
        if CORPUS_PATH.exists():
            txt_files = list(CORPUS_PATH.glob('*.txt'))
            st.write(f"**Files Found:** {len(txt_files)}")