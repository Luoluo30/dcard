1. 環境變數設定
請在專案根目錄建立一個 .env 檔案，並填入你的 API 金鑰：

'''bash
GOOGLE_GEMINI_API_KEY=你的api_key

2. 安裝套件
本專案支援使用 pip 來管理依賴。

'''bash
# 若你使用 requirements.txt
pip install -r requirements.txt

3. 執行程式
請確保你的終端機位於專案根目錄，使用以下指令來啟動 Streamlit：

'''bash
cd src
streamlit run main.py -- --store_dir ../data/vector
