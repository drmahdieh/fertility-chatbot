import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# -----------------------------
# تنظیمات اولیه
# -----------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

PDF_PATH = "data/infertility_guide.pdf"

# -----------------------------
# بارگذاری و پردازش PDF
# -----------------------------
@st.cache_resource
def load_and_split_pdf(_pdf_path):
    loader = PyPDFLoader(_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

@st.cache_resource
def create_vectorstore(_chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(_chunks, embeddings)

# -----------------------------
# ایجاد LLM و زنجیره QA
# -----------------------------
@st.cache_resource
def create_qa_chain(_vectorstore):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        return_source_documents=True
    )

# -----------------------------
# اجرای برنامه Streamlit
# -----------------------------
st.title("🤖 چت‌بات مشاوره PDF - Llama 3")
st.write("این چت‌بات به PDF متصل است و از مدل Llama 3 استفاده می‌کند.")

chunks = load_and_split_pdf(PDF_PATH)
vectorstore = create_vectorstore(chunks)
qa_chain = create_qa_chain(vectorstore)

user_question = st.text_input("سوال خود را بپرسید:")

if user_question:
    with st.spinner("در حال پردازش..."):
        result = qa_chain.invoke({"query": user_question})
        st.subheader("پاسخ:")
        st.write(result["result"])

        # نمایش منابع
        st.subheader("منابع:")
        for doc in result["source_documents"]:
            st.write(f"- صفحه: {doc.metadata.get('page', 'نامشخص')}")
