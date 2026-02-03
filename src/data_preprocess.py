import json
import random
import requests
import os

def download_drcd_dev():
    """下載 DRCD 開發集資料"""
    url = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_dev.json"
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = os.path.join(data_dir, "DRCD_dev.json")
    
    if os.path.exists(filename):
        print(f"檔案 {filename} 已存在，直接讀取。")
        return filename
        
    print(f"正在下載 {filename} ...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print("下載完成。")
        return filename
    except Exception as e:
        print(f"下載失敗: {e}")
        return None

def process_drcd_data(file_path, max_paragraphs=500, val_sample_size=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        drcd_data = json.load(f)

    kb_documents = []       # 準備存入 MongoDB 的知識庫文件
    qac_candidates = []     # 潛在的 QAC 驗證題候選池 (每個段落只取1題)

    paragraph_count = 0
    
    # 1. 遍歷文章與段落，收集 500 個段落
    for article in drcd_data['data']:
        title = article.get('title', '')
        
        for p in article['paragraphs']:
            if paragraph_count >= max_paragraphs:
                break
            
            context_text = p['context']
            context_id = p['id']
            
            # --- 建立知識庫資料結構 (MongoDB Document) ---
            kb_doc = {
                "doc_id": context_id,
                "text": context_text,
                "metadata": {
                    "source": "DRCD_dev",
                    "title": title,
                    "split": "dev"
                },
                # 預留 embedding 欄位，之後實作時填入
                "embedding": None 
            }
            kb_documents.append(kb_doc)
            
            # --- 建立驗證題候選池 (QAC) ---
            # 邏輯：如果該段落有問題，隨機挑選「1題」作為代表，避免同一個段落佔據太多測試題
            if p['qas']:
                selected_qa = random.choice(p['qas'])
                
                # DRCD 的 answers 是一個 list，通常取第一個作為標準答案
                ground_truth = selected_qa['answers'][0]['text'] if selected_qa['answers'] else ""
                
                qac_item = {
                    "qid": selected_qa['id'],
                    "question": selected_qa['question'],
                    "ground_truth_answer": ground_truth,
                    "ground_truth_context": context_text, # 這裡直接存文字方便驗證，也可以存 ID 讓程式去查
                    "context_id": context_id
                }
                qac_candidates.append(qac_item)
            
            paragraph_count += 1
        
        if paragraph_count >= max_paragraphs:
            break

    print(f"已收集 {len(kb_documents)} 個段落作為知識庫。")
    print(f"已從中提取 {len(qac_candidates)} 個不重複段落的問題作為候選。")

    # 2. 從候選池中隨機挑選 50 題
    if len(qac_candidates) < val_sample_size:
        print(f"警告：候選問題不足 {val_sample_size} 題，將使用所有可用問題。")
        validation_set = qac_candidates
    else:
        validation_set = random.sample(qac_candidates, val_sample_size)

    print(f"已隨機挑選 {len(validation_set)} 題作為最終驗證集。")

    return kb_documents, validation_set

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"檔案已儲存: {filename}")

# --- 主執行區 ---
if __name__ == "__main__":
    # 1. 取得檔案
    file_path = download_drcd_dev()
    
    if file_path:
        # 2. 處理資料
        kb_docs, val_set = process_drcd_data(file_path, max_paragraphs=500, val_sample_size=50)
        
        # 3. 儲存結果
        save_to_json(kb_docs, os.path.join("data", "drcd_knowledge_base.json"))
        save_to_json(val_set, os.path.join("data", "drcd_validation_set.json"))
        
        # 4. 顯示範例 (確認格式用)
        print("\n--- 知識庫資料範例 (MongoDB) ---")
        print(json.dumps(kb_docs[0], ensure_ascii=False, indent=2))
        
        print("\n--- 驗證集資料範例 (QAC) ---")
        print(json.dumps(val_set[0], ensure_ascii=False, indent=2))