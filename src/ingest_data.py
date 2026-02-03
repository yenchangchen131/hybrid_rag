import json
import os
from db_manager import MongoDBManager

def load_json(filename):
    """讀取 JSON 檔案"""
    if not os.path.exists(filename):
        print(f"錯誤：找不到檔案 {filename}")
        return None
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_ingestion():
    # 檔案路徑 (相對於專案根目錄執行)
    data_file = os.path.join("data", "drcd_knowledge_base.json")
    
    # 1. 讀取資料
    print(f"正在讀取 {data_file} ...")
    docs = load_json(data_file)
    if not docs:
        return

    print(f"共讀取到 {len(docs)} 筆文件。")

    # 2. 連線 MongoDB
    mongo = MongoDBManager()
    try:
        mongo.connect()
    except Exception:
        print("無法連線至 MongoDB，請確認 Docker Container 是否已啟動。")
        return

    # 3. 寫入資料
    COLLECTION_NAME = "drcd_knowledge_base"
    
    # 先清空舊資料 (開發階段方使便測試)
    mongo.get_collection(COLLECTION_NAME).delete_many({})
    print(f"已清空 {COLLECTION_NAME} 舊資料。")

    # 插入新資料
    mongo.insert_documents(COLLECTION_NAME, docs)

    # 4. 建立索引
    mongo.create_vector_index(COLLECTION_NAME)
    mongo.create_text_index(COLLECTION_NAME, ["text", "metadata.title"])

    print("資料匯入與索引建立完成。")

if __name__ == "__main__":
    run_ingestion()
