import sys
import os

# 將當前目錄加入路徑以便匯入 src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.db_manager import MongoDBManager

def drop_database():
    print("準備刪除資料庫...")
    
    # 初始化 Manager (預設 db_name='hybrid_rag')
    mongo = MongoDBManager()
    
    try:
        mongo.connect()
        print(f"成功連線至 MongoDB: {mongo.host}:{mongo.port}")
    except Exception as e:
        print(f"連線失敗: {e}")
        return

    target_db = mongo.db_name
    
    # 二次確認
    print(f"\n⚠️  警告: 即將永久刪除資料庫 '{target_db}'！")
    print("此操作無法復原。")
    confirm = input(f"請輸入 '{target_db}' 以確認刪除: ")
    
    if confirm == target_db:
        mongo.client.drop_database(target_db)
        print(f"\n✅ 資料庫 '{target_db}' 已成功刪除。")
    else:
        print("\n❌ 輸入不符，取消操作。")

if __name__ == "__main__":
    drop_database()
