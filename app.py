import os
from dotenv import load_dotenv
from openai import OpenAI
from pdf_reader import extract_text_from_pdf

openai.api_key = "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"
# بارگذاری کلید API از فایل env
#load_dotenv()
#api_key = os.getenv("OPENROUTER_API_KEY")

# اتصال به OpenRouter
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# بارگذاری متن از فایل PDF
pdf_path = "data/infertility_guide.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

# تابع برای پرسیدن سوال از مدل
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

# رابط ساده ترمینالی
while True:
    question = input("❓ سوال خود را وارد کنید (یا exit): ")
    if question.lower() == "exit":
        break
    answer = ask_bot(question)
    print("🤖 پاسخ:", answer)

