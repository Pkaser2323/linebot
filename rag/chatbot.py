import os
import time
import json
from dotenv import load_dotenv
from flask import Flask, request
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import requests
import mimetypes
# Load environment variables

from linebot.models import FlexSendMessage
from flexMessage import generate_flex_message  # 確保這行導入你的 Flex Message 生成函式


load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
LINE_ACCESS_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_SECRET = os.environ.get("LINE_SECRET")

# Configure Gemini AI
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# Sentence embedding model
EMBED_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v2"


def generate_retriever():
    print("Loading vector DB...")
    model_kwargs = {"device": "cuda"}
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs=model_kwargs)
    db = FAISS.load_local("vector_DB/diabetic_vector_db", embedding, allow_dangerous_deserialization=True)
    print("Done loading vector DB!")
    return db.as_retriever(search_kwargs={"k": 5})


retriever = generate_retriever()


def search_related_content(retriever, query):
    docs = retriever.invoke(query)
    return "\n---\n".join([doc.page_content for doc in docs])


def generate_answer(query, related_context):
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

    response = model.generate_content(template)
    return response.text if response else "不好意思，我不清楚。"

import io
import base64
import logging
from PIL import Image
from FatSecret.FatAPI import search_food_with_fatsecret
import re
def clean_markdown(text):
    """
    去除 Gemini AI 生成的 Markdown 標記，例如 **加粗**、*斜體*
    """
    return re.sub(r"[\*\_]", "", text).strip()


def extract_food_names_english(text):
    """
    從 Gemini Vision 生成的描述中擷取 **英文** 的食物名稱，回傳 list
    """
    extraction_prompt = f"""請從以下文字中找出 **所有主要的食物名稱（英文）**：
{text}

**輸出格式：**
- 只回傳英文食物名稱，不要其他描述或多餘的詞彙。
- 如果有多個食物，請用逗號分隔，例如：「apple, banana, sandwich」。
- 例如：
  「圖片顯示一個漢堡和薯條」 → 「burger, fries」
  「這是一碗白飯和一塊雞肉」 → 「rice, chicken」
"""

    response = model.generate_content(extraction_prompt)

    if not response or not hasattr(response, "text"):
        logging.error("Gemini AI 未能擷取食物名稱")
        return None

    # 取得 AI 回傳的文字並拆分為 list
    food_text = response.text.strip()
    food_list = [food.strip().lower() for food in food_text.split(",")]

    return food_list if food_list else None


from flexMessage import generate_flex_message  # 確保導入你的 Flex Message 生成函式

def analyze_food_with_gemini(image_path):
    """
    1️⃣ 使用 Gemini Vision 擷取 **英文** 食物名稱
    2️⃣ 使用 FatSecret API 查詢營養資訊
    3️⃣ 使用 Gemini 解析營養數據（英文），然後翻譯成繁體中文
    4️⃣ 產生 LINE Flex Message JSON
    """
    try:
        # **讀取圖片並轉換為 Base64**
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image_format = image.format
            image.save(buffered, format=image_format)
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # **Gemini Vision 分析圖片內容**
        vision_prompt = """請擷取圖片中 所有主要的食物名稱（英文），用逗號分隔，例如：
"apple, banana, sandwich"
"""

        vision_response = model.generate_content([
            {"mime_type": f"image/{image_format.lower()}", "data": image_base64},
            vision_prompt
        ])

        # **檢查 Vision AI 回應**
        if not vision_response or not hasattr(vision_response, "text"):
            logging.error("Gemini Vision AI 未回傳有效的結果")
            return "⚠️ 無法辨識圖片，請試試另一張！"

        food_list = [food.strip().lower() for food in vision_response.text.strip().split(",")]

        if not food_list:
            return "⚠️ 無法識別主要食物，請提供更清晰的圖片！"

        logging.info(f"🔍 提取的食物名稱: {food_list}")

        # **查詢 FatSecret API 並分析**
        flex_messages = []
        for food in food_list:
            nutrition_data = search_food_with_fatsecret(food)

            # 確保 API 回傳的數據是字典
            if not isinstance(nutrition_data, dict):
                logging.error(f"FatSecret API 回傳錯誤數據: {nutrition_data}")
                continue  # 跳過錯誤數據

            food_chinese_name = translate_to_chinese(food.capitalize())
            analysis_data = analyze_nutrition_with_gemini(nutrition_data)  # **回傳 JSON**
            flex_message = generate_flex_message(food_chinese_name, analysis_data, nutrition_data)
            flex_messages.append(flex_message)

        if not flex_messages:
            return "⚠️ 無法取得任何營養資訊"

        return {"type": "carousel", "contents": flex_messages}  # **回傳 Flex Message JSON**

    except Exception as e:
        logging.error(f"🚨 圖片分析時發生錯誤: {str(e)}")
        return "⚠️ 無法分析圖片，請稍後再試。"


import json
import logging


def analyze_nutrition_with_gemini(nutrition_text):
    """
    使用 Gemini AI 解析 FatSecret 的營養數據，確保返回 JSON，包含 優點、潛在風險、建議
    """
    analysis_prompt = f"""任務:
1. 你是一位專業營養師，請根據以下食物的營養資訊進行分析。
2. **輸出格式限制**: 只能輸出 JSON，不要額外的標題或文字，確保格式正確。
3. **JSON 結構如下**:
{{
  "優點": ["..."],
  "潛在風險": ["..."],
  "建議": ["..."]
}}

【營養數據】：
{nutrition_text}
請確保 **只輸出 JSON**，不要加任何額外的標題或描述。
"""

    response = model.generate_content(analysis_prompt)

    if not response or not hasattr(response, "text"):
        logging.error("Gemini AI 未回傳分析結果")
        return {"優點": [], "潛在風險": [], "建議": []}  # 返回空 JSON

    try:
        # 嘗試解析 AI 回應的 JSON
        json_response = json.loads(response.text.strip())

        # 確保回傳的 JSON 內容符合預期
        if isinstance(json_response, dict) and "優點" in json_response:
            return json_response

        logging.error("Gemini AI 回傳的 JSON 結構錯誤")
        return {"優點": [], "潛在風險": [], "建議": []}  # 預設空資料

    except json.JSONDecodeError:
        logging.error(f"Gemini AI 回傳的 JSON 格式錯誤，回應內容: {response.text}")
        return {"優點": [], "潛在風險": [], "建議": []}  # 預設空資料


def translate_to_chinese(english_text):
    """
    翻譯分析結果為繁體中文
    """
    translation_prompt = f"""請將以下內容翻譯為繁體中文，精準翻譯，只回傳食物名稱，不要其他描述或多餘的詞彙。
{english_text}
"""

    response = model.generate_content(translation_prompt)

    if not response or not hasattr(response, "text"):
        logging.error("Gemini AI 未回傳翻譯結果")
        return "⚠️ 無法翻譯分析結果，請稍後再試。"

    return response.text.strip()


# Flask app setup
app = Flask(__name__)
port = 5000
public_url = ngrok.connect(port).public_url
print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\" ")

# LINE Bot setup
line_bot_api = LineBotApi(LINE_ACCESS_TOKEN)
handler = WebhookHandler(LINE_SECRET)


@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)

        event = json_data['events'][0]
        tk = event['replyToken']
        message_type = event['message']['type']

        if message_type == "image":
            # 獲取圖片 ID
            image_id = event['message']['id']

            # 下載圖片
            image_url = f"https://api-data.line.me/v2/bot/message/{image_id}/content"
            headers = {"Authorization": f"Bearer {LINE_ACCESS_TOKEN}"}
            response = requests.get(image_url, headers=headers, stream=True)

            if response.status_code == 200:
                image_path = f"temp_{image_id}.jpg"
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content():
                        f.write(chunk)

                # 傳給 Gemini AI 進行分析
                food_result = analyze_food_with_gemini(image_path)

                if isinstance(food_result, str):
                    reply_message = TextSendMessage(text=food_result)
                else:
                    # 解析出 Flex Message 格式
                    flex_message = FlexSendMessage(alt_text="營養資訊", contents=food_result)
                    reply_message = flex_message

            else:
                reply_message = TextSendMessage(text="無法下載圖片，請稍後再試。")

        else:
            # 預設處理文字訊息
            msg = event['message']['text']
            related_context = search_related_content(retriever, msg)
            response = generate_answer(msg, related_context)
            reply_message = TextSendMessage(response)

        # 送出回應
        line_bot_api.reply_message(tk, reply_message)

    except Exception as e:
        print("Error:", str(e))
        print("Raw Body:", body)
    return 'OK'
from linebot.models import TextSendMessage

from linebot.models import TextMessage

def handle_postback(event):
    data = event.postback.data  # 取得 postback 資料

    if data.startswith("full_nutrition:"):
        food_name = data.split(":")[1]  # 取得食物名稱

        # **查詢完整營養資訊**
        nutrition_data = search_food_with_fatsecret(food_name)

        # **格式化回應**
        full_nutrition_message = f"""
📊 {food_name} 的完整營養資訊
🔥 卡路里: {nutrition_data.get('calories', 'N/A')} kcal
🍞 碳水化合物: {nutrition_data.get('carbohydrate', 'N/A')} g
🍬 糖分: {nutrition_data.get('sugar', 'N/A')} g
🍗 蛋白質: {nutrition_data.get('protein', 'N/A')} g
🥑 脂肪: {nutrition_data.get('fat', 'N/A')} g
🌾 纖維: {nutrition_data.get('fiber', 'N/A')} g
🧂 鈉: {nutrition_data.get('sodium', 'N/A')} mg
        """.strip()

        # **回覆完整營養資訊**
        reply_to_user(event.reply_token, TextMessage(text=full_nutrition_message))
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest

# 建立 MessagingApi 物件
messaging_api = MessagingApi(LINE_ACCESS_TOKEN)

def reply_to_user(reply_token, messages):
    """
    使用新版 LINE SDK (v3) 來回覆訊息
    """
    if isinstance(messages, list):
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=messages
            )
        )
    else:
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[messages]
            )
        )


if __name__ == "__main__":
    app.run(port=port)
