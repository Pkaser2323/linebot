import os
import time
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# Sentence embedding model
EMBED_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v2"

# Gemini model settings
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    "gemini-1.5-flash", safety_settings, generation_config=generation_config
)


def generate_retriever():
    print("Loading vector DB...")
    model_kwargs = {"device": "cuda"}
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs=model_kwargs)
    db = FAISS.load_local("vector_DB/diabetic_vector_db", embedding, allow_dangerous_deserialization=True)
    print("Done loading vector DB!\n")
    return db.as_retriever(search_kwargs={"k": 5})


def search_related_content(retriever, query):
    docs = retriever.invoke(query)
    return "\n---\n".join([doc.page_content for doc in docs])


def generate_answer(query, related_context, tokens):
    template = f"""
任務: 
1. 你是一位在台灣的糖尿病領域的專業護理師，需要以親切的口吻回答病患的問題。
2. 你必須依照下方的「相關文本」，再透過同義的改寫生成出最後的答案。
3. 輸出限制： 最多60字、只能使用繁體中文、純文字格式
4. 如果「相關文本」中有答案，一定要引用「相關文本」來回答；如果判斷與「病患的提問」沒有關連只要回答「不好意思，我不清楚。」即可。
------
「相關文本」：
{related_context}
------
「病患的提問」：
{query}
"""

    print(template)
    start_time = time.time()
    response = model.generate_content(template)
    qa_result = response.text
    end_time = time.time()

    print(f"Execution Time: {end_time - start_time}")
    tokens.append([
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.candidates_token_count
    ])

    return qa_result, tokens


# Initialize retriever
retriever = generate_retriever()

# Define questions
questions = [
    "為什麼糖尿病患者不敢吃水果？",
    "糖尿病患者該怎麼吃水果比較好？",
    "每天建議吃多少水果？",
    "水果的升糖指數會被哪些因素影響？",
    "一份水果大概是多少？",
    "紅豆是什麼種類的食物？",
    "喝無糖紅豆湯會對血糖有什麼影響？",
    "什麼情況下建議患者改用胰島素？",
    "哪些糖尿病患者需要直接打胰島素？",
    "打胰島素會不會導致需要洗腎？",
    "有哪些方法可以更完整地記錄飲食？",
    "記錄飲食時哪些資訊最重要？",
    "為什麼記錄飲食時，醣量是關鍵指標？",
    "熱量對於哪些人來說是關鍵的指標？",
    "記錄飲食時容易遇到哪些問題？",
    "烹調方式會怎麼影響飲食控制？",
    "氣喘如何治療？"
]

# Generate answers
answers = []
tokens = []

for query in questions:
    related_context = search_related_content(retriever, query)
    result, tokens = generate_answer(query, related_context, tokens)
    answers.append(result)
    time.sleep(5)

# Print results
for q, a, t in zip(questions, answers, tokens):
    print(q, a)
