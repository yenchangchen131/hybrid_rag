import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Generator:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def generate_answer(self, query, contexts):
        """
        根據檢索到的內容生成答案
        """
        # 組合 Context
        context_str = ""
        for i, ctx in enumerate(contexts):
            text = ctx['doc']['text']
            source = ctx['doc']['metadata']['title']
            context_str += f"[{i+1}] (出處: {source}): {text}\n\n"
            
        prompt = f"""你是一個專業的繁體中文問答助手。請根據以下的參考資訊回答使用者的問題。
如果參考資訊不足以回答，請說不知道。請勿編造事實。

參考資訊：
{context_str}

使用者問題：
{query}

請提供完整且有條理的回答：
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成答案時發生錯誤: {e}"
