import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Infertility Guide Chatbot", layout="wide")
st.title("🤖 چت‌بات راهنمای ناباروری")

# بارگذاری PDF
@st.cache_resource
def load_pdf(_pdf_path):
    loader = PyPDFLoader(_pdf_path)
    pages = loader.load()
    return pages

# تکه‌تکه کردن متن‌ها
@st.cache_resource
def split_pages(_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_pages)
    return chunks

# ساخت وکتوراستور
@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(_chunks, embeddings)
    return db

# ساخت مدل QA
@st.cache_resource
def create_qa_chain(_db):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        huggingfacehub_api_token=os.getenv("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")
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_db.as_retriever()
    )
    return qa_chain

# اجرای مراحل
pdf_path = "data/infertility_guide.pdf"
pages = load_pdf(pdf_path)
chunks = split_pages(pages)
db = create_vectorstore(chunks)
qa = create_qa_chain(db)

# رابط کاربری
question = st.text_input("سوال خود را از راهنمای ناباروری بپرسید:")
if question:
    with st.spinner("در حال پاسخ‌گویی..."):
        result = qa.run(question)
        st.markdown("### ✅ پاسخ:")
        st.write(result)
