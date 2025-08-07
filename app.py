import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

# عنوان صفحه
st.set_page_config(page_title="چت‌بات ناباروری با LLaMA3", layout="wide")
st.title("🤖 چت‌بات ناباروری با مدل LLaMA3")

# مسیر PDF
pdf_path = "data/infertility_guide.pdf"

# بارگذاری فایل PDF
@st.cache_data
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    return chunks

# ساخت بردارها
@st.cache_resource
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# ساخت مدل LLaMA3
def get_llama_model():
    return HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    )

# اجرای اصلی
chunks = load_and_process_pdf(pdf_path)
vectorstore = create_vectorstore(chunks)
llm = get_llama_model()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# رابط کاربری
question = st.text_input("سوال خود را وارد کنید:")
if question:
    with st.spinner("در حال پردازش..."):
        answer = qa_chain.run(question)
        st.success("پاسخ:")
        st.write(answer)
