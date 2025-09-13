
from langchain_core.document_loaders import Blob
# LangChain Community loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_core.documents import Document

import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import streamlit as st
import tempfile
from langchain_chroma import Chroma

# Handle asyncio loop for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------------------
# Utility functions
# ---------------------------
def text_injection(filepath): 
    loader = TextLoader(filepath)
    return loader.load()

def pdf_injection(filepath):
    blob = Blob.from_path(filepath)
    parser = PyPDFParser()
    return parser.parse(blob)

def chunk_text(doc):   
    chunker = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    return chunker.split_documents(doc)

# ---------------------------
# Frontend Design
# ---------------------------
st.set_page_config(page_title="Resume Evaluation Bot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🤖 AI Resume Evaluation Bot</h1>
    <p style='text-align: center;'>Upload your resume and compare it with a job description.<br>
    Get instant scoring, strengths, gaps, and shortlist recommendation.</p>
    """,
    unsafe_allow_html=True
)

# Layout: Resume upload | Job Description
col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("📄 Upload Resume", type=["txt", "pdf", "docx"])

with col2:
    query = st.text_area("💼 Paste Job Description", height=300)

# ---------------------------
# Processing
# ---------------------------
if st.button("🚀 Evaluate Resume", use_container_width=True):
    if file is None or query.strip() == "":
        st.error("⚠️ Please upload a resume and enter a job description.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tempfile_obj:
            tempfile_obj.write(file.read())
            filepath = tempfile_obj.name

        # Load documents
        if file.type == 'text/plain':
            doc = text_injection(filepath)
        elif file.type == 'application/pdf':
            doc = pdf_injection(filepath)
        else:
            st.error("❌ Unsupported file format. Please upload TXT or PDF.")
            st.stop()
        
        # Embedding model
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            timeout=120
        )

        chunks = chunk_text(doc)

        # Vector DB
        current=os.getcwd()
        db = Chroma.from_documents(
            doc,
            model,
            persist_directory=os.path.join(current, "chroma_db")
        )

        '''
        # Docker PGVector connection
        connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=model,
            connection=connection
        )'''

        # LLM model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0
        )
        
        @chain
        def RAG(query):
            retriever = db.as_retriever(search_kwargs={"k": 2})
            context = retriever.invoke(query)

            prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful assistant. Compare the candidate's resume with the job description.

                Resume: {context}
                Job Description: {question}

                Instructions for AI:
                1. Compare the candidate's resume to the job description.
                2. Identify relevant skills, experience, and qualifications.
                3. Score suitability on a scale of 1 (poor) to 10 (excellent).
                4. List the candidate's strengths that match the job.
                5. List gaps or missing skills.
                6. Decide whether the candidate should be shortlisted (Yes/No).

                Output format:
                Score: <number 1-10>
                Strengths: ["strength1", ..., "strengthN"]
                Gaps: ["gap1", ..., "gapN"]
                Shortlist: "Yes" or "No"
                """
            )

            chain = prompt | llm
            user_input = {"context": context, "question": query}
            response = chain.invoke(user_input)
            return response.content

        with st.spinner("⏳ Evaluating resume..."):
            res = RAG.invoke(query)
            
        
        st.markdown("## 📊 Evaluation Result")
        st.success(res)
