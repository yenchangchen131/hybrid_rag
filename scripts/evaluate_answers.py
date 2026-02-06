#!/usr/bin/env python
"""
LLM 語意評估腳本

使用 GPT-4o-mini 評估模型回答與標準答案的語意一致性。

使用方式：
    uv run python scripts/evaluate_answers.py
    uv run python scripts/evaluate_answers.py --input data/rag_results_hybrid.json
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from tqdm import tqdm
from openai import OpenAI

from core.config import settings


EVALUATION_PROMPT = """請判斷「模型回答」是否與「標準答案」語意一致。

問題：{question}
標準答案：{gold_answer}
模型回答：{model_answer}

判斷標準：
- 如果模型回答包含標準答案的核心資訊，且沒有明顯錯誤，請回答 "Pass"
- 如果模型回答與標準答案語意不一致、有錯誤、或完全無關，請回答 "Fail"

請只回答 "Pass" 或 "Fail"，不要有任何其他文字。"""


def evaluate_answer(
    client: OpenAI,
    question: str,
    gold_answer: str,
    model_answer: str,
    model: str = "gpt-4o-mini",
) -> tuple[str, bool]:
    """使用 LLM 評估答案
    
    Returns:
        (raw_response, is_pass)
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        model_answer=model_answer,
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        
        raw = response.choices[0].message.content.strip()
        is_pass = raw.lower() == "pass"
        return raw, is_pass
        
    except Exception as e:
        print(f"⚠️ 評估失敗: {e}")
        return f"Error: {e}", False


def evaluate_all(results: list[dict], client: OpenAI) -> list[dict]:
    """評估所有結果"""
    evaluated = []
    
    for r in tqdm(results, desc="語意評估中"):
        raw_response, is_pass = evaluate_answer(
            client=client,
            question=r["question"],
            gold_answer=r.get("gold_answer", ""),
            model_answer=r.get("generated_answer", ""),
        )
        
        evaluated.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "question_type": r.get("question_type", "unknown"),
            "source_dataset": r.get("source_dataset", "unknown"),
            "gold_answer": r.get("gold_answer", ""),
            "generated_answer": r.get("generated_answer", ""),
            "llm_judgment": raw_response,
            "is_pass": is_pass,
        })
    
    return evaluated


def calculate_pass_rate(evaluated: list[dict]) -> dict:
    """計算通過率"""
    total = len(evaluated)
    passed = sum(1 for e in evaluated if e["is_pass"])
    
    # 按問題類型分組
    by_type = {}
    for e in evaluated:
        q_type = e["question_type"]
        if q_type not in by_type:
            by_type[q_type] = {"total": 0, "passed": 0}
        by_type[q_type]["total"] += 1
        if e["is_pass"]:
            by_type[q_type]["passed"] += 1
    
    # 按資料來源分組
    by_source = {}
    for e in evaluated:
        source = e["source_dataset"]
        if source not in by_source:
            by_source[source] = {"total": 0, "passed": 0}
        by_source[source]["total"] += 1
        if e["is_pass"]:
            by_source[source]["passed"] += 1
    
    return {
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 4) if total > 0 else 0,
        },
        "by_question_type": {
            k: {**v, "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] > 0 else 0}
            for k, v in by_type.items()
        },
        "by_source": {
            k: {**v, "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] > 0 else 0}
            for k, v in by_source.items()
        },
    }


def print_results(stats: dict) -> None:
    """輸出結果"""
    summary = stats["summary"]
    
    print("\n" + "=" * 60)
    print("📊 語意評估結果")
    print("=" * 60)
    print(f"總題數:     {summary['total']}")
    print(f"通過數:     {summary['passed']}")
    print(f"Pass Rate:  {summary['pass_rate']:.2%}")
    
    print("\n" + "-" * 60)
    print("📈 按問題類型")
    for q_type, data in stats["by_question_type"].items():
        print(f"  【{q_type}】 {data['pass_rate']:.2%} ({data['passed']}/{data['total']})")
    
    print("\n" + "-" * 60)
    print("📚 按資料來源")
    for source, data in stats["by_source"].items():
        print(f"  【{source}】 {data['pass_rate']:.2%} ({data['passed']}/{data['total']})")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LLM 語意評估")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/rag_results_hybrid.json",
        help="RAG 結果檔案路徑"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="輸出檔案路徑 (預設: 根據 input 自動命名)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="評估用 LLM 模型 (預設: gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 自動命名輸出檔案
    if args.output:
        output_path = Path(args.output)
    else:
        input_name = input_path.stem
        mode_suffix = input_name.replace("rag_results_", "")
        output_path = input_path.parent / f"answer_evaluation_{mode_suffix}.json"
    
    print("=" * 60)
    print("🔬 LLM 語意評估")
    print("=" * 60)
    print(f"📂 輸入: {input_path}")
    print(f"🤖 模型: {args.model}")
    
    # 載入結果
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", data)
    print(f"   共 {len(results)} 筆待評估")
    
    # 初始化 OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # 評估
    evaluated = evaluate_all(results, client)
    
    # 計算統計
    stats = calculate_pass_rate(evaluated)
    
    # 輸出
    print_results(stats)
    
    # 儲存
    output_data = {
        "metadata": {
            "input_file": str(input_path),
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "results": evaluated,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 詳細結果已儲存至: {output_path}")


if __name__ == "__main__":
    main()
