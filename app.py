import os
from openai import OpenAI
import streamlit as st
from pdf_reader import extract_text_from_pdf

# کلید API را مستقیماً وارد کردیم
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

# متن PDF فقط یک‌بار بارگذاری شود
@st.cache_data
def load_pdf_text():
    pdf_path = "data/infertility_guide.pdf"
    return extract_text_from_pdf(pdf_path)

pdf_text = load_pdf_text()

def ask_bot(user_question):
    messages = [
        {"role": "system", "content": "شما یک دستیار پزشکی هستید که فقط بر اساس اطلاعات زیر از فایل پاسخ می‌دهد:\n" + pdf_text[:4000]},
        {"role": "user", "content": user_question}
    ]
    
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=messages
    )
    
    return response.choices[0].message.content


# رابط گرافیکی با Streamlit
st.title("🤖 چت‌بات ناباروری")
st.write("سوالی درباره راهنمای ناباروری دارید؟ بپرسید:")

user_input = st.text_input("❓ سوال خود را بنویسید:")

if st.button("ارسال سوال"):
    if user_input.strip() != "":
        with st.spinner("در حال پردازش..."):
            answer = ask_bot(user_input)
            st.markdown("### 🤖 پاسخ:")
            st.write(answer)
    else:
        st.warning("لطفاً یک سوال وارد کنید.")
