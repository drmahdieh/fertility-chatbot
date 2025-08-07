import streamlit as st
from openai import OpenAI
from pdf_reader import extract_text_from_pdf

# تنظیم کلید API به صورت مستقیم (امن نیست، فقط برای تست مناسب است)
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

# بارگذاری متن از فایل PDF
pdf_path = "data/infertility_guide.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# رابط کاربری استریملت
st.set_page_config(page_title="دستیار ناباروری", layout="wide")
st.title("🤖 دستیار مبتنی بر هوش مصنوعی برای راهنمایی ناباروری")

user_question = st.text_input("❓ سوال خود را وارد کنید:")

if user_question:
    messages = [
        {"role": "system", "content": "شما یک دستیار پزشکی هستید که فقط بر اساس اطلاعات زیر از فایل پاسخ می‌دهد:\n" + pdf_text[:4000]},
        {"role": "user", "content": user_question}
    ]
    
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=messages
    )
    
    st.markdown("### 🧠 پاسخ")
    st.write(response.choices[0].message.content)
