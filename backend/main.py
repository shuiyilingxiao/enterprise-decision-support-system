from pathlib import Path
import json
import re
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from conditions import CONDITION_NAMES, short_name

API_KEY = "123456"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"

PROMPT_FILE = Path(__file__).parent / "prompt.txt"
PLACEHOLDER = "（在这里写你的提示词"

app = FastAPI()


class Req(BaseModel):
    message: str
    mode: str = "match"
    sub_mode: str | None = None
    conversation_id: str | None = None
    history: list | None = None


def parse_model_json(raw: str) -> dict:
    """从模型返回里尽量抠出 JSON，失败返回 {}"""
    if not raw:
        return {}

    s = raw.strip()

    # 去掉 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()

    # 截取第一个 { 到最后一个 }
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        s = s[start:end + 1]

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def format_hits(hits: list[str]) -> str:
    """把条件 ID 列表拼成给用户看的 Markdown"""
    if not hits:
        return "我这次没有识别到明确的条件。能再具体说说你的处境吗？比如资金、市场、团队、渠道、政策方面。"

    lines = ["我识别到以下条件：\n"]
    for cid in hits:
        name = CONDITION_NAMES.get(cid)
        if name:
            lines.append(f"- **{cid}** {name}")
        else:
            lines.append(f"- **{cid}**（未知条件，请检查 prompt 输出）")
    lines.append("\n这些对吗？有没有要补充或更正的？")
    return "\n".join(lines)


@app.post("/api/analyze")
def analyze(req: Req):
    # 1. 读提示词
    prompt_tpl = PROMPT_FILE.read_text(encoding="utf-8").strip() if PROMPT_FILE.exists() else ""
    if not prompt_tpl or PLACEHOLDER in prompt_tpl:
        return {"reply": "提示词未配置，请先在 prompt.txt 里写提示词。"}

    prompt = prompt_tpl.replace("{{MESSAGE}}", req.message)

    # 2. 调模型
    try:
        r = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3. 解析 JSON
    parsed = parse_model_json(raw)
    hits = parsed.get("hits", [])

    # 过滤不在条件库里的 ID
    valid_hits = [h for h in hits if h in CONDITION_NAMES]

    # 4. 拼回复
    reply = format_hits(valid_hits)

    # 5. 解析失败时，把原始返回附上，方便你调 prompt
    if not parsed:
        reply += f"\n\n---\n**模型原始返回（供调试）**：\n```\n{raw[:500]}\n```"

    return {
        "reply": reply,
        "hits": valid_hits,
        "suggestions": ["对，继续", "还有要补充的", "重新描述"] if valid_hits else [],
    }


app.mount("/", StaticFiles(directory="public", html=True), name="static")