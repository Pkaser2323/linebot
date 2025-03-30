import os
import time
import json
from dotenv import load_dotenv
from flask import Flask, request
from pyngrok import ngrok
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, FlexSendMessage, PostbackEvent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import requests
import mimetypes
import io
import base64
import logging
from PIL import Image
from FatSecret.FatAPI import search_food_with_fatsecret
import re
from flexMessage import generate_flex_message, generate_calorie_source_flex_message
# Load environment variables
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

# 添加全局變量來保存查詢結果
global_data_store = {}

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
1. 你是一位在台灣的糖尿病領域的專業護理師，需要以專業嚴謹但親切的態度回答病患的問題。

2. 請仔細分析下方的「相關文本」，並按照以下步驟回答：
   a. 從「相關文本」中提取可靠且相關的醫療資訊
   b. 確保所提供的每一項建議都有文獻依據
   c. 整合資訊時，需明確區分：
      - 確定的醫療建議（有明確依據）
      - 一般性建議（基於專業知識）
   d. 使用準確的醫療術語，並提供清晰的解釋

3. 回答要求：
   - 字數限制：最多60字，且回答須為純文本
   - 不要回答文獻依據，只回答病患的問題
   - 使用繁體中文
4. 回答限制：
   - 如果相關文本中沒有足夠的專業依據，必須明確告知：「這個問題需要更多專業資訊才能完整回答，建議您諮詢主治醫師」
   - 不進行推測性回答
   - 對於可能影響病患安全的建議，必須提醒諮詢醫療人員

------
「相關文本」：
{related_context}
------
「病患的提問」：
{query}
"""

    response = model.generate_content(template)
    return response.text if response else "不好意思，我不清楚。"

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


def analyze_food_with_gemini(image_path):
    """
    1️⃣ 使用 Gemini Vision 擷取 **英文** 食物名稱
    2️⃣ 使用 FatSecret API 查詢營養資訊
    3️⃣ 使用 Gemini 解析營養數據（英文），然後翻譯成繁體中文
    4️⃣ 輸出 簡潔且易讀 的結果
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
            return TextSendMessage(text="⚠️ 無法辨識圖片，請試試另一張！")

        food_list = [food.strip().lower() for food in vision_response.text.strip().split(",")]

        if not food_list:
            return TextSendMessage(text="⚠️ 無法識別主要食物，請提供更清晰的圖片！")

        logging.info(f"🔍 提取的食物名稱: {food_list}")

        # **查詢 FatSecret API 並分析**
        all_results = []
        nutrition_data_list = []
        food_chinese_names = []
        api_data_found = False  # 標記是否找到API數據
        food_english_names = []  # 保存成功查詢到的英文食物名稱，用於後續詳細信息查詢
        
        for food in food_list:
            nutrition_data = search_food_with_fatsecret(food)

            # 確保 API 回傳的數據是字典
            if not isinstance(nutrition_data, dict):
                logging.error(f"FatSecret API 回傳錯誤數據: {nutrition_data}")
                continue  # 跳過錯誤數據

            food_chinese_name = translate_to_chinese(food.capitalize())
            food_chinese_names.append(food_chinese_name)
            
            # 添加食物名稱到營養數據中，用於後續處理
            nutrition_data['food_name'] = food
            nutrition_data['food_chinese_name'] = food_chinese_name
            food_english_names.append(food)
            
            # 分析並取得優點、風險、建議
            analysis_data = analyze_nutrition_for_flex(nutrition_data)
            
            # 保存營養數據，供後續生成 FlexMessage 使用
            nutrition_data_list.append(nutrition_data)
            
            # 檢查是否找到有效的API數據
            if 'calories' in nutrition_data and nutrition_data.get('calories'):
                api_data_found = True

            # **格式化輸出**
            formatted_result = f"""
📊 {food_chinese_name} 的營養資訊
🔥 卡路里: {nutrition_data.get('calories', 'N/A')} kcal
🍞 碳水化合物: {nutrition_data.get('carbohydrate', 'N/A')} g
🍗 蛋白質: {nutrition_data.get('protein', 'N/A')} g
🥑 脂肪: {nutrition_data.get('fat', 'N/A')} g
🍬 糖: {nutrition_data.get('sugar', 'N/A')} g
🌾 纖維: {nutrition_data.get('fiber', 'N/A')} g
🧂 鈉: {nutrition_data.get('sodium', 'N/A')} mg
"""
            all_results.append(formatted_result.strip())

        # 計算熱量來源佔比
        calorie_sources = calculate_calorie_sources(nutrition_data_list)
        
        # 檢查營養數據是否為空
        if not nutrition_data_list:
            return TextSendMessage(text="⚠️ 無法獲取食物的營養資訊，請稍後再試。")
        
        # 將查詢狀態添加到全局數據存儲中，用於詳細頁面顯示
        global_data_store[','.join(food_english_names)] = {
            'api_data_found': api_data_found,
            'nutrition_data_list': nutrition_data_list,
            'food_chinese_names': food_chinese_names
        }
        
        # 更新熱量來源數據，添加API數據標記
        calorie_sources['is_estimated'] = not api_data_found
        
        # 生成熱量來源分析的 FlexMessage
        flex_message = generate_calorie_source_flex_message(food_chinese_names, calorie_sources)
        
        # 確保返回的是 LINE 的消息對象
        if isinstance(flex_message, dict):
            # 如果是字典，轉換為 FlexSendMessage
            return FlexSendMessage(alt_text=f"{food_chinese_names[0]} 的熱量來源分析", contents=flex_message)
        else:
            # 如果已經是 FlexSendMessage 或其他 LINE 消息對象，直接返回
            return flex_message

    except Exception as e:
        logging.error(f"🚨 圖片分析時發生錯誤: {str(e)}")
        return TextSendMessage(text="⚠️ 無法分析圖片，請稍後再試。")

def analyze_nutrition_for_flex(nutrition_data):
    """
    分析營養數據，提取優點、風險和建議，以便生成 FlexMessage
    """
    analysis_prompt = f"""任務:
1. 你是一位專業營養師，請根據以下食物的營養資訊進行分析：
2. 分析結果必須包含這三個區塊：優點、潛在風險、建議（針對糖尿病患者）
3. 每個區塊提供 1-2 點簡潔的分析，每點不超過15字
4. 使用繁體中文

【營養數據】：
{nutrition_data}

請用以下JSON格式回答：
{{"優點":["優點1", "優點2"], "潛在風險":["風險1", "風險2"], "建議":["建議1", "建議2"]}}
"""

    try:
        gemini_response = model.generate_content(analysis_prompt)
        if not gemini_response or not hasattr(gemini_response, "text"):
            return {"優點": [], "潛在風險": [], "建議": []}
            
        # 解析 JSON 格式的回應
        analysis_text = gemini_response.text.strip()
        # 確保只提取 JSON 部分
        match = re.search(r'(\{.*\})', analysis_text, re.DOTALL)
        if match:
            analysis_json = match.group(1)
            try:
                return json.loads(analysis_json)
            except:
                return {"優點": [], "潛在風險": [], "建議": []}
        return {"優點": [], "潛在風險": [], "建議": []}
    except Exception as e:
        print(f"分析營養數據時出錯: {str(e)}")
        return {"優點": [], "潛在風險": [], "建議": []}

def calculate_calorie_sources(nutrition_data_list):
    """
    計算熱量來源佔比（碳水化合物、蛋白質、脂肪、糖分）
    """
    total_carb_calories = 0
    total_protein_calories = 0
    total_fat_calories = 0
    total_sugar_calories = 0  # 新增糖分熱量計算
    total_calories = 0
    
    # 熱量換算：碳水4卡/克，蛋白質4卡/克，脂肪9卡/克，糖分4卡/克
    for data in nutrition_data_list:
        carb = float(data.get('carbohydrate', 0) or 0)
        protein = float(data.get('protein', 0) or 0)
        fat = float(data.get('fat', 0) or 0)
        sugar = float(data.get('sugar', 0) or 0)  # 獲取糖分含量
        
        carb_cal = carb * 4
        protein_cal = protein * 4
        fat_cal = fat * 9
        sugar_cal = sugar * 4  # 糖分熱量計算（同碳水化合物）
        
        total_carb_calories += carb_cal
        total_protein_calories += protein_cal
        total_fat_calories += fat_cal
        total_sugar_calories += sugar_cal  # 累加糖分熱量
        total_calories += float(data.get('calories', 0) or 0)
    
    # 計算佔比
    if total_calories > 0:
        carb_percentage = (total_carb_calories / total_calories) * 100
        protein_percentage = (total_protein_calories / total_calories) * 100
        fat_percentage = (total_fat_calories / total_calories) * 100
        sugar_percentage = (total_sugar_calories / total_calories) * 100  # 計算糖分比例
    else:
        # 如果沒有熱量資訊，使用大語言模型尋找建議值
        food_names = []
        for data in nutrition_data_list:
            if 'food_name' in data and data['food_name']:
                food_names.append(data['food_name'])
        
        # 如果有食物名稱，使用大語言模型估算
        if food_names:
            estimated_values = estimate_nutrition_with_gemini(food_names)
            total_calories = estimated_values.get('total_calories', 100)
            total_carb_calories = estimated_values.get('carbs_calories', 50)
            total_protein_calories = estimated_values.get('protein_calories', 20)
            total_fat_calories = estimated_values.get('fat_calories', 30)
            total_sugar_calories = estimated_values.get('sugar_calories', 10)
        else:
            # 若無法獲取食物名稱，使用預設值
            total_calories = 100
            total_carb_calories = 50
            total_protein_calories = 20
            total_fat_calories = 30
            total_sugar_calories = 10
    
    return {
        "carbs_calories": round(total_carb_calories, 0),  # 改為直接返回熱量值而非百分比
        "protein_calories": round(total_protein_calories, 0),
        "fat_calories": round(total_fat_calories, 0),
        "sugar_calories": round(total_sugar_calories, 0),  # 添加糖分熱量值
        "total_calories": round(total_calories, 0),
        "is_estimated": total_calories == 0  # 添加標記，表示是否為估算值
    }

def estimate_nutrition_with_gemini(food_names):
    """
    使用Gemini獲取食物的估計營養成分
    
    Args:
        food_names: 食物名稱列表
    
    Returns:
        包含估計營養值的字典
    """
    # 組合所有食物名稱
    food_list = "、".join(food_names)
    
    # 構建提示詞
    prompt = f"""請根據營養學知識，估算以下食物的大致熱量來源分佈：{food_list}

請提供以下信息的估計值：
1. 總熱量（大卡）
2. 碳水化合物熱量（大卡）
3. 蛋白質熱量（大卡）
4. 脂肪熱量（大卡）
5. 糖分熱量（大卡）

請使用以下JSON格式回應：
{{"total_calories": 數值, "carbs_calories": 數值, "protein_calories": 數值, "fat_calories": 數值, "sugar_calories": 數值}}

注意：這些只是估計值，非精確數據。
"""
    
    try:
        # 呼叫Gemini模型
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, "text"):
            return {"total_calories": 100, "carbs_calories": 50, "protein_calories": 20, "fat_calories": 30, "sugar_calories": 10}
            
        # 從回應中提取JSON
        result_text = response.text.strip()
        # 尋找JSON部分
        match = re.search(r'(\{.*\})', result_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                estimated_values = json.loads(json_str)
                # 確保所有必要的鍵存在
                required_keys = ["total_calories", "carbs_calories", "protein_calories", "fat_calories", "sugar_calories"]
                for key in required_keys:
                    if key not in estimated_values:
                        estimated_values[key] = 0
                return estimated_values
            except:
                # JSON解析失敗，返回預設值
                return {"total_calories": 100, "carbs_calories": 50, "protein_calories": 20, "fat_calories": 30, "sugar_calories": 10}
        
        # 未找到有效JSON，返回預設值
        return {"total_calories": 100, "carbs_calories": 50, "protein_calories": 20, "fat_calories": 30, "sugar_calories": 10}
    except Exception as e:
        print(f"估算營養值時出錯: {str(e)}")
        return {"total_calories": 100, "carbs_calories": 50, "protein_calories": 20, "fat_calories": 30, "sugar_calories": 10}

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
        
        # 處理 Postback 事件
        if event['type'] == 'postback':
            data = event['postback']['data']
            
            # 處理「查看完整熱量來源分析」按鈕
            if data.startswith('detailed_calorie_source:'):
                food_names = data.split(':', 1)[1].split(',')
                
                # 檢查是否有對應的英文食物名稱
                found = False
                for key in global_data_store.keys():
                    key_foods = key.split(',')
                    # 如果中文名與英文名的數量一致，則嘗試匹配
                    if len(key_foods) == len(food_names):
                        stored_chinese_names = global_data_store[key].get('food_chinese_names', [])
                        if all(name in stored_chinese_names for name in food_names):
                            # 找到匹配的英文鍵
                            food_key = key
                            found = True
                            break
                            
                # 如果沒找到匹配的英文鍵，則直接使用中文名
                if not found:
                    food_key = ','.join(food_names)
                    
                detailed_analysis = generate_detailed_nutrition_flex(food_names, food_key)
                reply_message = detailed_analysis
            else:
                reply_message = TextSendMessage(text="抱歉，無法處理此請求")
                
            line_bot_api.reply_message(tk, reply_message)
            return 'OK'
        
        # 處理一般訊息事件
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
                result = analyze_food_with_gemini(image_path)
                
                # 檢查結果類型並回應
                if isinstance(result, str):
                    reply_message = TextSendMessage(text=result)
                else:
                    # 回傳 FlexMessage
                    reply_message = result
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

def generate_detailed_nutrition_flex(food_names, food_key=None):
    """
    生成詳細營養資訊的 FlexMessage
    
    Args:
        food_names: 食物名稱列表
        food_key: 用於在global_data_store中查找的鍵
    """
    # 從全局存儲獲取數據，如果有API數據則使用，否則使用估算值
    if food_key is None:
        food_key = ','.join(food_names)
        
    stored_data = global_data_store.get(food_key, {})
    
    # 檢查是否有已保存的API數據
    if stored_data and stored_data.get('api_data_found', False):
        # 使用存儲的API數據
        nutrition_data_list = stored_data.get('nutrition_data_list', [])
        food_chinese_names = stored_data.get('food_chinese_names', food_names)
        is_estimated = False
        
        # 從API數據中提取詳細營養信息
        detailed_nutrition = extract_detailed_nutrition_from_api(nutrition_data_list)
    else:
        # 使用大語言模型獲取估算的詳細營養資訊
        detailed_nutrition = estimate_detailed_nutrition_with_gemini(food_names)
        food_chinese_names = food_names
        is_estimated = True
    
    # 創建食物清單字串
    if len(food_chinese_names) > 1:
        food_title = "、".join(food_chinese_names[:3])
        if len(food_chinese_names) > 3:
            food_title += f" 等{len(food_chinese_names)}種食物"
    else:
        food_title = food_chinese_names[0] if food_chinese_names else "食物"
    
    # 從營養數據中提取值
    total_calories = detailed_nutrition.get('total_calories', 0)
    
    # 碳水相關數據
    carbs = detailed_nutrition.get('carbs', {})
    carbs_total = carbs.get('total', 0)
    carbs_calories = carbs.get('calories', 0)
    carbs_percent = carbs.get('percent', 0)
    sugar = carbs.get('sugar', {})
    sugar_total = sugar.get('total', 0)
    sugar_calories = sugar.get('calories', 0)
    fiber = carbs.get('fiber', {})
    fiber_total = fiber.get('total', 0)
    fiber_calories = fiber.get('calories', 0)
    
    # 蛋白質相關數據
    protein = detailed_nutrition.get('protein', {})
    protein_total = protein.get('total', 0)
    protein_calories = protein.get('calories', 0)
    protein_percent = protein.get('percent', 0)
    
    # 脂肪相關數據
    fat = detailed_nutrition.get('fat', {})
    fat_total = fat.get('total', 0)
    fat_calories = fat.get('calories', 0)
    fat_percent = fat.get('percent', 0)
    saturated_fat = fat.get('saturated', {})
    saturated_fat_total = saturated_fat.get('total', 0)
    saturated_fat_calories = saturated_fat.get('calories', 0)
    unsaturated_fat = fat.get('unsaturated', {})
    unsaturated_fat_total = unsaturated_fat.get('total', 0)
    unsaturated_fat_calories = unsaturated_fat.get('calories', 0)
    
    # 其他營養素
    sodium = detailed_nutrition.get('sodium', 0)
    potassium = detailed_nutrition.get('potassium', 0)
    cholesterol = detailed_nutrition.get('cholesterol', 0)
    
    # 創建詳細營養資訊的 FlexMessage
    contents = {
        "type": "bubble",
        "size": "mega",
        "header": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": f"{food_title} 詳細營養資訊",
                    "weight": "bold",
                    "color": "#1DB446",
                    "size": "lg"
                }
            ],
            "backgroundColor": "#F9F9F9"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                # 如果是估算值，添加提示信息
                {
                    "type": "text",
                    "text": "⚠️ 以下為AI估算數據，僅供參考" if is_estimated else "🔍 以下為FatSecret營養資料庫數據",
                    "color": "#FF6B6E" if is_estimated else "#1DB446",
                    "size": "xs",
                    "margin": "sm",
                    "align": "start",
                    "weight": "bold"
                },
                {
                    "type": "text",
                    "text": "熱量來源詳細分析",
                    "weight": "bold",
                    "size": "md",
                    "margin": "md"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "營養素",
                                    "size": "sm",
                                    "color": "#555555",
                                    "weight": "bold",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": "數量",
                                    "size": "sm",
                                    "color": "#555555",
                                    "weight": "bold",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": "佔總熱量",
                                    "size": "sm",
                                    "color": "#555555",
                                    "weight": "bold",
                                    "align": "end",
                                    "flex": 3
                                }
                            ]
                        },
                        {
                            "type": "separator",
                            "margin": "md"
                        },
                        # 碳水化合物
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "碳水化合物",
                                    "size": "sm",
                                    "color": "#0066cc",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{carbs_total} g",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{carbs_calories} 大卡 (約{carbs_percent}%)",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "md"
                        },
                        # 其中: 糖分
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "  • 糖分",
                                    "size": "xs",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{sugar_total} g",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{sugar_calories} 大卡",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "sm"
                        },
                        # 其中: 膳食纖維
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "  • 膳食纖維",
                                    "size": "xs",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{fiber_total} g",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{fiber_calories} 大卡",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "sm"
                        },
                        # 蛋白質
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "蛋白質",
                                    "size": "sm",
                                    "color": "#cc6600",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{protein_total} g",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{protein_calories} 大卡 (約{protein_percent}%)",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "lg"
                        },
                        # 脂肪
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "脂肪",
                                    "size": "sm",
                                    "color": "#336633",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{fat_total} g",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{fat_calories} 大卡 (約{fat_percent}%)",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "lg"
                        },
                        # 其中: 飽和脂肪
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "  • 飽和脂肪",
                                    "size": "xs",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{saturated_fat_total} g",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{saturated_fat_calories} 大卡",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "sm"
                        },
                        # 其中: 不飽和脂肪
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "  • 不飽和脂肪",
                                    "size": "xs",
                                    "color": "#999999",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": f"約{unsaturated_fat_total} g",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"{unsaturated_fat_calories} 大卡",
                                    "size": "xs",
                                    "color": "#999999",
                                    "align": "end",
                                    "flex": 3
                                }
                            ],
                            "margin": "sm"
                        },
                        {
                            "type": "separator",
                            "margin": "lg"
                        },
                        # 總計
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "總計",
                                    "size": "sm",
                                    "color": "#111111",
                                    "weight": "bold",
                                    "flex": 3
                                },
                                {
                                    "type": "text",
                                    "text": "-",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "center",
                                    "flex": 2
                                },
                                {
                                    "type": "text",
                                    "text": f"約{total_calories} 大卡",
                                    "size": "sm",
                                    "color": "#FF4500",
                                    "align": "end",
                                    "weight": "bold",
                                    "flex": 3
                                }
                            ],
                            "margin": "lg"
                        }
                    ],
                    "backgroundColor": "#FFFFFF",
                    "cornerRadius": "md",
                    "paddingAll": "md"
                },
                {
                    "type": "separator",
                    "margin": "xxl"
                },
                {
                    "type": "text",
                    "text": "其他營養素",
                    "weight": "bold",
                    "size": "md",
                    "margin": "xl"
                },
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "lg",
                    "contents": [
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "鈉",
                                    "size": "sm",
                                    "color": "#555555",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": f"約{sodium} mg",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 1
                                }
                            ]
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "鉀",
                                    "size": "sm",
                                    "color": "#555555",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": f"約{potassium} mg",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 1
                                }
                            ],
                            "margin": "md"
                        },
                        {
                            "type": "box",
                            "layout": "horizontal",
                            "contents": [
                                {
                                    "type": "text",
                                    "text": "膽固醇",
                                    "size": "sm",
                                    "color": "#555555",
                                    "flex": 1
                                },
                                {
                                    "type": "text",
                                    "text": f"約{cholesterol} mg",
                                    "size": "sm",
                                    "color": "#111111",
                                    "align": "end",
                                    "flex": 1
                                }
                            ],
                            "margin": "md"
                        }
                    ],
                    "backgroundColor": "#FFFFFF",
                    "cornerRadius": "md",
                    "paddingAll": "md"
                }
            ],
            "paddingAll": "13px",
            "backgroundColor": "#F9F9F9"
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "* 所有數據" + ("均為AI估算值，僅供參考" if is_estimated else "來自FatSecret營養資料庫") + "，實際值可能因食物種類、品牌和烹飪方式而異",
                    "size": "xs",
                    "color": "#aaaaaa",
                    "wrap": True,
                    "align": "center"
                }
            ],
            "backgroundColor": "#F9F9F9"
        },
        "styles": {
            "header": {
                "separator": True
            },
            "footer": {
                "separator": True
            }
        }
    }
    
    return FlexSendMessage(alt_text=f"{food_title} 詳細營養資訊", contents=contents)

def extract_detailed_nutrition_from_api(nutrition_data_list):
    """
    從API獲取的營養數據提取詳細營養信息
    
    Args:
        nutrition_data_list: API數據列表
    
    Returns:
        整合後的詳細營養信息
    """
    # 初始化累計值
    total_calories = 0
    total_carbs = 0
    total_protein = 0
    total_fat = 0
    total_sugar = 0
    total_fiber = 0
    total_sodium = 0
    total_potassium = 0
    total_cholesterol = 0
    
    # 累加所有食物的營養素
    for data in nutrition_data_list:
        total_calories += float(data.get('calories', 0) or 0)
        total_carbs += float(data.get('carbohydrate', 0) or 0)
        total_protein += float(data.get('protein', 0) or 0)
        total_fat += float(data.get('fat', 0) or 0)
        total_sugar += float(data.get('sugar', 0) or 0)
        total_fiber += float(data.get('fiber', 0) or 0)
        total_sodium += float(data.get('sodium', 0) or 0)
        total_potassium += float(data.get('potassium', 0) or 0)
        total_cholesterol += float(data.get('cholesterol', 0) or 0)
    
    # 計算熱量
    carb_calories = total_carbs * 4
    protein_calories = total_protein * 4
    fat_calories = total_fat * 9
    sugar_calories = total_sugar * 4
    fiber_calories = total_fiber * 2
    
    # 計算每種營養素佔總熱量的百分比
    total_energy_calories = carb_calories + protein_calories + fat_calories
    
    if total_energy_calories > 0:
        carbs_percent = round((carb_calories / total_energy_calories) * 100)
        protein_percent = round((protein_calories / total_energy_calories) * 100)
        fat_percent = round((fat_calories / total_energy_calories) * 100)
    else:
        carbs_percent = 0
        protein_percent = 0
        fat_percent = 0
    
    # 估算飽和脂肪和不飽和脂肪（通常API沒有提供這些數據，需要估算）
    saturated_fat = round(total_fat * 0.3, 1)  # 假設30%為飽和脂肪
    unsaturated_fat = round(total_fat * 0.7, 1)  # 假設70%為不飽和脂肪
    
    saturated_fat_calories = round(saturated_fat * 9, 1)
    unsaturated_fat_calories = round(unsaturated_fat * 9, 1)
    
    # 構建詳細營養數據
    return {
        "total_calories": round(total_calories, 0),
        "carbs": {
            "total": round(total_carbs, 1),
            "calories": round(carb_calories, 0),
            "percent": carbs_percent,
            "sugar": {
                "total": round(total_sugar, 1),
                "calories": round(sugar_calories, 0)
            },
            "fiber": {
                "total": round(total_fiber, 1),
                "calories": round(fiber_calories, 0)
            }
        },
        "protein": {
            "total": round(total_protein, 1),
            "calories": round(protein_calories, 0),
            "percent": protein_percent
        },
        "fat": {
            "total": round(total_fat, 1),
            "calories": round(fat_calories, 0),
            "percent": fat_percent,
            "saturated": {
                "total": saturated_fat,
                "calories": saturated_fat_calories
            },
            "unsaturated": {
                "total": unsaturated_fat,
                "calories": unsaturated_fat_calories
            }
        },
        "sodium": round(total_sodium, 0),
        "potassium": round(total_potassium, 0),
        "cholesterol": round(total_cholesterol, 0)
    }

def estimate_detailed_nutrition_with_gemini(food_names):
    """
    使用Gemini獲取食物的詳細估計營養成分
    
    Args:
        food_names: 食物名稱列表
    
    Returns:
        包含詳細估計營養值的字典
    """
    # 組合所有食物名稱
    food_list = "、".join(food_names)
    
    # 構建提示詞
    prompt = f"""請根據營養學知識，估算以下食物的詳細營養成分：{food_list}

請提供以下信息的估計值，並使用指定JSON格式回應：
{{
  "total_calories": 總熱量（大卡）,
  "carbs": {{
    "total": 碳水化合物總克數,
    "calories": 碳水熱量（大卡）,
    "percent": 佔總熱量百分比,
    "sugar": {{
      "total": 糖分克數,
      "calories": 糖分熱量（大卡）
    }},
    "fiber": {{
      "total": 膳食纖維克數,
      "calories": 膳食纖維熱量（大卡）
    }}
  }},
  "protein": {{
    "total": 蛋白質總克數,
    "calories": 蛋白質熱量（大卡）,
    "percent": 佔總熱量百分比
  }},
  "fat": {{
    "total": 脂肪總克數,
    "calories": 脂肪熱量（大卡）,
    "percent": 佔總熱量百分比,
    "saturated": {{
      "total": 飽和脂肪克數,
      "calories": 飽和脂肪熱量（大卡）
    }},
    "unsaturated": {{
      "total": 不飽和脂肪克數,
      "calories": 不飽和脂肪熱量（大卡）
    }}
  }},
  "sodium": 鈉含量（毫克）,
  "potassium": 鉀含量（毫克）,
  "cholesterol": 膽固醇含量（毫克）
}}

注意：這些只是估計值，非精確數據。請盡量合理估算每種營養素的數值，考慮指定食物的一般份量。
"""
    
    try:
        # 呼叫Gemini模型
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, "text"):
            return get_default_detailed_nutrition()
            
        # 從回應中提取JSON
        result_text = response.text.strip()
        # 尋找JSON部分
        match = re.search(r'(\{.*\})', result_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                estimated_values = json.loads(json_str)
                # 檢查主要結構是否存在
                if not all(key in estimated_values for key in ["total_calories", "carbs", "protein", "fat"]):
                    return get_default_detailed_nutrition()
                return estimated_values
            except:
                # JSON解析失敗，返回預設值
                return get_default_detailed_nutrition()
        
        # 未找到有效JSON，返回預設值
        return get_default_detailed_nutrition()
    except Exception as e:
        print(f"估算詳細營養值時出錯: {str(e)}")
        return get_default_detailed_nutrition()

def get_default_detailed_nutrition():
    """提供預設的詳細營養資訊結構"""
    return {
        "total_calories": 100,
        "carbs": {
            "total": 15,
            "calories": 60,
            "percent": 60,
            "sugar": {
                "total": 5,
                "calories": 20
            },
            "fiber": {
                "total": 2,
                "calories": 0
            }
        },
        "protein": {
            "total": 5,
            "calories": 20,
            "percent": 20
        },
        "fat": {
            "total": 2,
            "calories": 20,
            "percent": 20,
            "saturated": {
                "total": 0.5,
                "calories": 4.5
            },
            "unsaturated": {
                "total": 1.5,
                "calories": 15.5
            }
        },
        "sodium": 120,
        "potassium": 200,
        "cholesterol": 5
    }

# 註冊事件處理函數
@handler.add(PostbackEvent)
def handle_postback(event):
    # 這個函數不需要實現內容，因為我們已經在上面的 linebot 函數中處理了 postback
    pass

if __name__ == "__main__":
    app.run(port=port)