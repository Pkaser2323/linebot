from pypdf import PdfReader

def check_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        print(f'\n檢查文件：{pdf_path}')
        print(f'總頁數：{len(reader.pages)}')
        print('第一頁內容預覽：\n')
        print(reader.pages[0].extract_text()[:500])
        print('\n' + '='*50 + '\n')
        return True
    except Exception as e:
        print(f'讀取 {pdf_path} 時發生錯誤：{str(e)}')
        return False

# 檢查兩個 PDF 文件
pdfs = [
    'rag/docs/diabetic_acticles.pdf',
    'rag/docs/diabetic_qa.pdf'
]

for pdf in pdfs:
    check_pdf(pdf) 