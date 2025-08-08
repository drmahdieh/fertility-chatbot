import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# تنظیمات اولیه
# -----------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# -----------------------------
# بارگذاری vectorstore آماده
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# ایجاد زنجیره پرسش و پاسخ
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
# رابط کاربری Streamlit
# -----------------------------
st.set_page_config(page_title="چت‌بات PDF", page_icon="🤖")
st.title("🤖 چت‌بات مشاوره PDF - Llama 3")
st.write("سوالت رو درباره فایل PDF بپرس، من با مدل Llama 3 پاسخ می‌دم.")

# بارگذاری وکتور آماده
vectorstore = load_vectorstore()
qa_chain = create_qa_chain(vectorstore)

# دریافت سوال کاربر
user_question = st.text_input("سوال خود را بپرسید:")

# پاسخ‌دهی
if user_question:
    with st.spinner("در حال پردازش..."):
        result = qa_chain.invoke({"query": user_question})
        st.subheader("پاسخ:")
        st.write(result["result"])

        st.subheader("منابع:")
        for doc in result["source_documents"]:
            st.write(f"- صفحه: {doc.metadata.get('page', 'نامشخص')}")
