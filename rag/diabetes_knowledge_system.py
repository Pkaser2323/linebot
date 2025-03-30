import pandas as pd
import os
import time
import re
import torch
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import google.generativeai as genai

# 載入環境變數
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("警告：未設定 GOOGLE_API_KEY 環境變數！")
else:
    genai.configure(api_key=API_KEY)

# 檢查是否有 GPU 可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用設備: {device}")

# 嵌入模型設定
EMBED_MODEL_NAME = "DMetaSoul/sbert-chinese-general-v2"

# Gemini 模型設定
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 0.3,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}


class DiabetesKnowledgeSystem:
    def __init__(self, rebuild_db=False):
        """初始化糖尿病知識系統
        
        Args:
            rebuild_db: 是否重建向量資料庫
        """
        self.db_path = "vector_DB/diabetic_vector_db"
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash", safety_settings, generation_config=generation_config
        )
        
        # 檢查是否需要重建資料庫
        if rebuild_db or not os.path.exists(self.db_path):
            print("重建向量資料庫...")
            self.build_vector_database()
        
        # 初始化檢索器
        self.retriever = self.generate_retriever()
    
    def load_article_data(self, file_path):
        """載入文章資料 (標題、內文)"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"錯誤: 無法讀取 {file_path}: {e}")
            return []
        
        documents = []
        
        # 檢查欄位名稱 - 標準格式: 標題, 內文
        if '標題' in df.columns and '內文' in df.columns:
            title_col = '標題'
            content_col = '內文'
        # 其他可能的欄位名稱格式
        elif 'title' in df.columns.str.lower() and 'content' in df.columns.str.lower():
            title_col = df.columns[df.columns.str.lower() == 'title'][0]
            content_col = df.columns[df.columns.str.lower() == 'content'][0]
        elif 'title' in df.columns.str.lower() and 'article' in df.columns.str.lower():
            title_col = df.columns[df.columns.str.lower() == 'title'][0]
            content_col = df.columns[df.columns.str.lower() == 'article'][0]
        else:
            print(f"警告: {file_path} 缺少必要的欄位 (標題/title, 內文/content/article)")
            return []
        
        print(f"使用欄位: {title_col}, {content_col}")
        
        for idx, row in df.iterrows():
            if isinstance(row[title_col], str) and isinstance(row[content_col], str):
                title = row[title_col].strip()
                content = row[content_col].strip()
                if title and content:  # 確保有內容
                    # 合併標題和內容
                    text = f"標題: {title}\n內容: {content}"
                    documents.append(Document(page_content=text, metadata={"source": "articles", "id": idx, "title": title}))
        
        return documents

    def load_qa_data(self, file_path):
        """載入問答資料 (標題、問題、回答)"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"錯誤: 無法讀取 {file_path}: {e}")
            return []
        
        documents = []
        
        # 檢查欄位名稱 - 標準格式: 標題, 問題, 回答
        if '標題' in df.columns and '問題' in df.columns and '回答' in df.columns:
            title_col = '標題'
            question_col = '問題'
            answer_col = '回答'
        # 其他可能的欄位名稱格式
        elif any('title' in col.lower() for col in df.columns) and any('question' in col.lower() for col in df.columns) and any('answer' in col.lower() for col in df.columns):
            title_col = [col for col in df.columns if 'title' in col.lower()][0]
            question_col = [col for col in df.columns if 'question' in col.lower()][0]
            answer_col = [col for col in df.columns if 'answer' in col.lower()][0]
        # 檢查是否有問題和回答，但沒有標題
        elif any('question' in col.lower() for col in df.columns) and any('answer' in col.lower() for col in df.columns):
            title_col = None
            question_col = [col for col in df.columns if 'question' in col.lower()][0]
            answer_col = [col for col in df.columns if 'answer' in col.lower()][0]
            print(f"注意: {file_path} 沒有標題欄位，將使用問題作為標題")
        else:
            print(f"警告: {file_path} 缺少必要的欄位 (標題/title 和/或 問題/question, 回答/answer)")
            return []
        
        columns_used = [col for col in [title_col, question_col, answer_col] if col is not None]
        print(f"使用欄位: {', '.join(columns_used)}")
        
        for idx, row in df.iterrows():
            title = row[title_col].strip() if title_col and isinstance(row[title_col], str) else ""
            
            # 如果沒有標題，使用問題作為標題
            if not title and title_col is None and isinstance(row[question_col], str):
                title = row[question_col][:50]  # 使用問題的前50個字作為標題
                if len(row[question_col]) > 50:
                    title += "..."
            
            # 處理問題
            if isinstance(row[question_col], str):
                question = row[question_col].strip()
                if question:  # 確保問題有內容
                    # 合併標題和問題
                    text = f"標題: {title}\n問題: {question}"
                    documents.append(Document(page_content=text, metadata={"source": "questions", "id": idx, "title": title}))
            
            # 處理回答
            if isinstance(row[answer_col], str):
                answer = row[answer_col].strip()
                if answer:  # 確保回答有內容
                    # 合併標題和回答
                    text = f"標題: {title}\n回答: {answer}"
                    documents.append(Document(page_content=text, metadata={"source": "answers", "id": idx, "title": title}))
        
        return documents

    def load_pdf_article(self, file_path):
        """載入 PDF 文章資料"""
        print(f"正在處理 PDF 文章: {file_path}")
        
        documents = []
        try:
            # 使用 PyPDFLoader 載入 PDF
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            
            # 獲取檔案名稱作為預設標題
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            for i, doc in enumerate(pdf_docs):
                # 從 PDF 頁面中提取內容
                content = doc.page_content.strip()
                
                # 如果有內容，創建文檔
                if content:
                    # 嘗試從內容中提取標題
                    lines = content.split('\n')
                    title = lines[0].strip() if lines and len(lines[0]) < 100 else f"{file_name} - 第 {i+1} 頁"
                    
                    # 創建文檔
                    text = f"標題: {title}\n內容: {content}"
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "source": "pdf_article", 
                            "id": f"{file_name}_p{i+1}",
                            "title": title,
                            "page": i+1,
                            "pdf_path": file_path
                        }
                    ))
            
            print(f"已從 PDF 載入 {len(documents)} 頁文章")
        except Exception as e:
            print(f"處理 PDF 時發生錯誤: {e}")
        
        return documents

    def load_pdf_qa(self, file_path):
        """載入 PDF 問答資料並嘗試識別問題和答案"""
        print(f"正在處理 PDF 問答: {file_path}")
        
        documents = []
        try:
            # 使用 PyPDFLoader 載入 PDF
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            
            # 獲取檔案名稱作為預設標題
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            for i, doc in enumerate(pdf_docs):
                # 從 PDF 頁面中提取內容
                content = doc.page_content.strip()
                
                if content:
                    # 嘗試將內容分割成問題和答案
                    qa_pairs = self.split_into_qa_pairs(content)
                    
                    if qa_pairs:
                        # 如果成功分割成問答對
                        for j, (question, answer) in enumerate(qa_pairs):
                            # 使用問題作為標題
                            title = question[:50]
                            if len(question) > 50:
                                title += "..."
                            
                            # 創建問題文檔
                            q_text = f"標題: {title}\n問題: {question}"
                            documents.append(Document(
                                page_content=q_text, 
                                metadata={
                                    "source": "pdf_questions", 
                                    "id": f"{file_name}_p{i+1}_q{j+1}",
                                    "title": title,
                                    "page": i+1,
                                    "pdf_path": file_path
                                }
                            ))
                            
                            # 創建回答文檔
                            a_text = f"標題: {title}\n回答: {answer}"
                            documents.append(Document(
                                page_content=a_text, 
                                metadata={
                                    "source": "pdf_answers", 
                                    "id": f"{file_name}_p{i+1}_a{j+1}",
                                    "title": title,
                                    "page": i+1,
                                    "pdf_path": file_path
                                }
                            ))
                    else:
                        # 如果無法分割成問答對，將整頁視為一篇文章
                        title = f"{file_name} - 第 {i+1} 頁"
                        text = f"標題: {title}\n內容: {content}"
                        documents.append(Document(
                            page_content=text, 
                            metadata={
                                "source": "pdf_article", 
                                "id": f"{file_name}_p{i+1}",
                                "title": title,
                                "page": i+1,
                                "pdf_path": file_path
                            }
                        ))
            
            print(f"已從 PDF 載入 {len(documents)} 項問答/文章內容")
        except Exception as e:
            print(f"處理 PDF 時發生錯誤: {e}")
        
        return documents

    def split_into_qa_pairs(self, text):
        """嘗試將文本分割為問答對"""
        qa_pairs = []
        
        # 尋找常見的問答模式
        qa_patterns = [
            # 標準問答模式
            (r'問[:：](.+?)答[:：](.+?)(?=問[:：]|$)', r'問\s*\d+[:：](.+?)答\s*\d+[:：](.+?)(?=問\s*\d+[:：]|$)'),
            # Q&A模式
            (r'Q[:：](.+?)A[:：](.+?)(?=Q[:：]|$)', r'Q\s*\d+[:：](.+?)A\s*\d+[:：](.+?)(?=Q\s*\d+[:：]|$)'),
            # 問：答：模式
            (r'問題[:：](.+?)回答[:：](.+?)(?=問題[:：]|$)', r'問題\s*\d+[:：](.+?)回答\s*\d+[:：](.+?)(?=問題\s*\d+[:：]|$)')
        ]
        
        for pattern_pair in qa_patterns:
            for pattern in pattern_pair:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    for match in matches:
                        question = match[0].strip()
                        answer = match[1].strip()
                        if question and answer:
                            qa_pairs.append((question, answer))
                    
                    # 如果找到了問答對，就返回
                    if qa_pairs:
                        return qa_pairs
        
        # 嘗試更寬鬆的匹配：尋找問號結尾的句子
        sentences = re.split(r'[。！？\n]+', text)
        i = 0
        while i < len(sentences) - 1:
            if sentences[i].strip().endswith('?') or sentences[i].strip().endswith('？'):
                question = sentences[i].strip()
                answer_parts = []
                j = i + 1
                # 收集後續句子作為答案，直到遇到下一個問句
                while j < len(sentences) and not (sentences[j].strip().endswith('?') or sentences[j].strip().endswith('？')):
                    answer_parts.append(sentences[j].strip())
                    j += 1
                
                if answer_parts:
                    answer = '。'.join(answer_parts)
                    qa_pairs.append((question, answer))
                    i = j - 1
            i += 1
        
        return qa_pairs

    def create_vector_db(self, documents, output_path):
        """創建向量資料庫"""
        print(f"建立向量資料庫，總文件數: {len(documents)}")
        
        # 初始化嵌入模型
        model_kwargs = {"device": device}
        embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME, 
            model_kwargs=model_kwargs
        )
        
        # 建立向量資料庫
        db = FAISS.from_documents(documents, embedding)
        
        # 保存向量資料庫
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        db.save_local(output_path)
        print(f"向量資料庫已保存至: {output_path}")
        
        return db

    def check_pdf_contains_qa(self, file_path):
        """檢查 PDF 是否主要包含問答內容"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # 取樣檢查前幾頁
            sample_pages = min(3, len(pages))
            qa_indicators = 0
            
            for i in range(sample_pages):
                content = pages[i].page_content.lower()
                
                # 檢查問答指標
                if re.search(r'問[:：]|答[:：]|問題[:：]|回答[:：]|q[:：]|a[:：]', content):
                    qa_indicators += 1
                
                # 檢查問號密度
                question_marks = content.count('?') + content.count('？')
                if question_marks > 3:  # 若一頁中有多個問號，可能是問答
                    qa_indicators += 1
            
            # 如果超過半數的採樣頁面表現出問答特徵
            return qa_indicators >= sample_pages
        except:
            return False  # 如果處理出錯，預設不是問答

    def build_vector_database(self):
        """建立向量資料庫"""
        # 設定輸入和輸出路徑
        # 文章資料路徑
        article_paths = [
            "cleaned_data/merged_article2s_cleaned.csv",
        ]
        
        # PDF 文章路徑
        pdf_article_paths = [
            "docs/diabetic_acticles.pdf",
        ]
        
        # 問答資料路徑
        qa_paths = [
            "cleaned_data/taiwan_ehospital_diabetes_qa_cleaned.csv",
        ]
        
        # PDF 問答路徑
        pdf_qa_paths = [
            "docs/diabetic_qa.pdf",
        ]
        
        # 載入所有文章資料
        article_docs = []
        for path in article_paths:
            if os.path.exists(path):
                docs = self.load_article_data(path)
                article_docs.extend(docs)
                print(f"已載入文章資料: {len(docs)} 篇 (來自 {path})")
            else:
                print(f"警告: 找不到文件 {path}")
        
        # 載入 PDF 文章
        for path in pdf_article_paths:
            if os.path.exists(path):
                # 檢查 PDF 是否主要是問答格式
                if self.check_pdf_contains_qa(path):
                    print(f"檢測到 {path} 主要包含問答內容，將使用問答載入器")
                    docs = self.load_pdf_qa(path)
                else:
                    docs = self.load_pdf_article(path)
                article_docs.extend(docs)
                print(f"已載入 PDF 文章資料: {len(docs)} 項 (來自 {path})")
            else:
                print(f"警告: 找不到文件 {path}")
        
        # 載入所有問答資料
        qa_docs = []
        for path in qa_paths:
            if os.path.exists(path):
                docs = self.load_qa_data(path)
                qa_docs.extend(docs)
                print(f"已載入問答資料: {len(docs)} 條 (來自 {path})")
            else:
                print(f"警告: 找不到文件 {path}")
        
        # 載入 PDF 問答
        for path in pdf_qa_paths:
            if os.path.exists(path):
                docs = self.load_pdf_qa(path)
                qa_docs.extend(docs)
                print(f"已載入 PDF 問答資料: {len(docs)} 條 (來自 {path})")
            else:
                print(f"警告: 找不到文件 {path}")
        
        # 合併所有文件
        all_docs = article_docs + qa_docs
        print(f"總文件數: {len(all_docs)} 篇")
        
        if len(all_docs) == 0:
            print("錯誤: 沒有找到任何有效文件，無法建立向量資料庫")
            return
        
        # 建立向量資料庫
        self.create_vector_db(all_docs, self.db_path)

    def generate_retriever(self):
        """生成檢索器"""
        print("載入向量資料庫...")
        model_kwargs = {"device": device}
        embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME, model_kwargs=model_kwargs)
        db = FAISS.load_local(self.db_path, embedding, allow_dangerous_deserialization=True)
        print("向量資料庫載入完成！\n")
        # 檢索 4 個最相關的文檔
        return db.as_retriever(search_kwargs={"k": 4})

    def search_related_content(self, query):
        """搜尋相關內容"""
        docs = self.retriever.invoke(query)
        # 根據相關性排序文檔
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
        return "\n---\n".join([doc.page_content for doc in sorted_docs])

    def generate_answer(self, query, related_context=None, verbose=False):
        """生成回答"""
        # 如果沒有提供相關上下文，則搜尋相關內容
        if related_context is None:
            related_context = self.search_related_content(query)
        
        template = f"""
任務: 
1. 你是一位在台灣的糖尿病領域的專業護理師，需要以專業且嚴謹的態度回答病患的問題。

2. 請仔細分析下方的「相關文本」，並按照以下步驟回答：
   a. 從「相關文本」中提取可靠且相關的醫療資訊
   b. 確保所提供的每一項建議都有文獻依據
   c. 整合資訊時，需明確區分：
      - 確定的醫療建議（有明確依據）
      - 一般性建議（基於專業知識）
   d. 使用準確的醫療術語，並提供清晰的解釋

3. 回答要求：
   - 字數限制：最多60字
   - 使用繁體中文
   - 分段呈現，重要醫療建議需加粗標記
   - 若提供數據，必須註明是否來自相關文本

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
        
        if verbose:
            print(template)
        
        start_time = time.time()
        response = self.model.generate_content(template)
        qa_result = response.text
        end_time = time.time()
        
        if verbose:
            print(f"執行時間: {end_time - start_time:.2f} 秒")
            print(f"輸入字符: {response.usage_metadata.prompt_token_count}, 輸出字符: {response.usage_metadata.candidates_token_count}")
        
        return qa_result

    def interactive_qa(self):
        """互動式問答"""
        print("=== 糖尿病知識問答系統 ===")
        print("輸入 'exit' 或 'quit' 結束對話\n")
        
        while True:
            query = input("\n請輸入您的問題: ")
            
            if query.lower() in ['exit', 'quit', '退出', '結束']:
                print("感謝使用，再見！")
                break
            
            print("\n正在思考...")
            try:
                answer = self.generate_answer(query)
                print("\n回答:")
                print(answer)
            except Exception as e:
                print(f"發生錯誤: {e}")

    def batch_qa(self, questions):
        """批次處理問答"""
        answers = []
        
        for query in questions:
            print(f"處理問題: {query}")
            try:
                related_context = self.search_related_content(query)
                answer = self.generate_answer(query, related_context)
                answers.append(answer)
            except Exception as e:
                print(f"處理問題時出錯: {e}")
                answers.append("處理出錯")
            
            # 避免超過API速率限制
            time.sleep(2)
        
        return answers


def main():
    """主函數"""
    print("初始化糖尿病知識系統...")
    
    # 詢問是否重建資料庫
    rebuild = input("是否重建向量資料庫？(y/n): ").lower() == 'y'
    
    # 初始化系統
    system = DiabetesKnowledgeSystem(rebuild_db=rebuild)
    
    # 啟動互動式問答
    system.interactive_qa()


if __name__ == "__main__":
    main() 