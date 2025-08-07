from openai import OpenAI
import PyPDF2

# کلید API خودت رو اینجا وارد کن
api_key = "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"

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

# تابع استخراج متن از PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# مسیر فایل PDF خودت رو اینجا قرار بده
pdf_path = "data/infertility_guide.pdf"

pdf_text = extract_text_from_pdf(pdf_path)

# برای اینکه زیاد نشه فقط 4000 کاراکتر اول رو می‌گیریم
pdf_text = pdf_text[:4000]

def ask_bot(question):
    messages = [
        {"role": "system", "content": "شما یک دستیار پزشکی هستید که فقط بر اساس اطلاعات زیر پاسخ می‌دهید:\n" + pdf_text},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("چت‌بات پزشکی شروع به کار کرد. برای خروج exit را تایپ کنید.")
    while True:
        user_question = input("سوال خود را بپرسید: ")
        if user_question.lower() == "exit":
            break
        answer = ask_bot(user_question)
        print("پاسخ:", answer)

