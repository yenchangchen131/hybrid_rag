"""
生成服務

封裝 OpenAI Chat Completion API，根據檢索結果生成答案。
"""

from openai import OpenAI

from core.config import settings
from models.response import RetrievalResult


class GenerationService:
    """LLM 生成服務"""
    
    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.LLM_MODEL
        self._temperature = settings.LLM_TEMPERATURE
    
    def generate_answer(
        self,
        query: str,
        contexts: list[RetrievalResult],
        max_contexts: int = 5,
    ) -> str:
        """根據檢索結果生成答案
        
        Args:
            query: 使用者問題
            contexts: 檢索結果
            max_contexts: 最大使用的上下文數量
            
        Returns:
            生成的答案
        """
        # 組合上下文
        context_str = ""
        for i, ctx in enumerate(contexts[:max_contexts]):
            source = ctx.original_source or "unknown"
            context_str += f"[{i+1}] (來源: {source}):\n{ctx.content}\n\n"
        
        prompt = f"""你是一個專業的繁體中文問答助手。請根據以下的參考資訊回答使用者的問題。
如果參考資訊不足以回答，請說「根據提供的資訊無法回答此問題」。請勿編造事實。

參考資訊：
{context_str}

使用者問題：
{query}

請提供完整且有條理的回答："""
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions in Traditional Chinese."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"生成答案時發生錯誤: {e}"
