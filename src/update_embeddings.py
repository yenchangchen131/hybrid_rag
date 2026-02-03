from db_manager import MongoDBManager
from embedding import get_embedding
from tqdm import tqdm

def update_embeddings():
    db = MongoDBManager()
    collection = db.get_collection("drcd_knowledge_base")
    
    # 找出所有還沒有 embedding 的文件
    # query = {"embedding": None} 
    # 或者為了確保全部更新，我們可以先只查一部分，這裡預設全部檢查
    cursor = collection.find({"embedding": None})
    total_docs = collection.count_documents({"embedding": None})
    
    print(f"共有 {total_docs} 筆文件需要生成 Embedding...")
    
    if total_docs == 0:
        print("所有文件已有 Embedding，無需更新。")
        return

    # 批次更新
    for doc in tqdm(cursor, total=total_docs):
        text = doc.get("text", "")
        if text:
            emb = get_embedding(text)
            if emb:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": emb}}
                )
    
    print("Embedding 更新完成。")

if __name__ == "__main__":
    update_embeddings()
