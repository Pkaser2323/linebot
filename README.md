# 糖尿病飲食分析 LINE Bot 🤖

這是一個基於 LINE Messaging API 開發的糖尿病飲食分析助手，能夠透過圖片辨識和營養資料庫分析食物的營養成分，並提供針對糖尿病患者的飲食建議。

## 功能特點 ✨

- 🔍 **圖片食物辨識**：使用 Google Gemini Vision AI 辨識食物
- 📊 **營養資訊分析**：整合 FatSecret 營養資料庫
- 🍎 **熱量來源分析**：顯示碳水化合物、蛋白質、脂肪等營養素佔比
- 🔬 **詳細營養成分**：提供糖分、膳食纖維、膽固醇等詳細資訊
- 💡 **智能建議**：根據營養分析提供糖尿病飲食建議
- 💬 **營養諮詢**：整合專業糖尿病營養知識庫

## 系統架構 🏗

```
project/
├── rag/                      # 主要程式碼目錄
│   ├── chatbot.py           # LINE Bot 主程式
│   ├── flexMessage.py       # LINE Flex Message 模板
│   └── FatAPI/             # FatSecret API 介接
├── vector_DB/               # 向量資料庫
├── .env                     # 環境變數配置
└── README.md               # 專案說明文件
```

## 技術堆疊 🛠

- **後端框架**：Flask
- **AI 模型與框架**：
  - Google Gemini 1.5 Flash (圖像識別)
  - DMetaSoul/sbert-chinese-general-v2 (文本嵌入)
  - Hugging Face Transformers (自然語言處理)
  - LangChain (AI 應用框架)
- **資料庫**：
  - FAISS 向量資料庫
  - FatSecret 營養資料庫
- **通訊平台**：LINE Messaging API
- **部署工具**：ngrok

## 環境設定 ⚙️

1. 安裝相依套件：
```bash
pip install -r requirements.txt
```

2. 設定環境變數：
```
GOOGLE_API_KEY=your_google_api_key
LINE_ACCESS_TOKEN=your_line_access_token
LINE_SECRET=your_line_secret
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

3. 啟動服務：
```bash
python rag/chatbot.py
```

## API 金鑰申請 🔑

1. **LINE Developers**：
   - 註冊 LINE Developers 帳號
   - 創建 Messaging API Channel
   - 取得 Channel Secret 和 Channel Access Token

2. **Google Cloud**：
   - 註冊 Google Cloud 帳號
   - 啟用 Gemini API
   - 創建 API 金鑰

3. **FatSecret Platform**：
   - 註冊 FatSecret Platform API 帳號
   - 取得 API 存取金鑰

## 使用說明 📱

1. 掃描 LINE Bot QR Code 加入好友
2. 拍攝或上傳食物照片
3. 等待系統分析並回傳：
   - 基本營養資訊
   - 熱量來源分析
   - 糖尿病飲食建議
4. 點選「查看詳細營養資訊」獲取更多資訊

