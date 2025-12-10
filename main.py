import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import whisper

# --------------------
# 1. 初始化 FastAPI
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境先放开，后面可以收紧
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# 2. 课程数据（和之前一样，写死一课）
# --------------------
LESSON_DB = {
    "intro_001": {
        "lesson_id": "intro_001",
        "title": "自我介绍 · 入门",
        "audio_url": "",
        "sentences": [
            {
                "id": "s1",
                "text": "Hola, soy Ana.",
                "translation": "你好，我是 Ana。"
            },
            {
                "id": "s2",
                "text": "Hoy vamos a practicar algunas frases básicas en español.",
                "translation": "今天我们来练习一些基础西班牙语句子。"
            },
            {
                "id": "s3",
                "text": "Primero, repite conmigo: ¿Cómo te llamas? Me llamo Ana.",
                "translation": "首先，跟我重复：你叫什么名字？我叫 Ana。"
            }
        ]
    }
}


@app.get("/lesson/{lesson_id}")
async def get_lesson(lesson_id: str):
    lesson = LESSON_DB.get(lesson_id)
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    return lesson


# --------------------
# 3. 初始化 Whisper 模型（只加载一次）
# --------------------
# 模型越小越快，tiny 就够你一句两三秒的西语练习用
print("Loading Whisper model... (tiny)")
whisper_model = whisper.load_model("tiny")
print("Whisper model loaded.")


# --------------------
# 4. DeepSeek 配置
# --------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # 通用聊天模型，便宜够用

if not DEEPSEEK_API_KEY:
    print("⚠️ WARNING: DEEPSEEK_API_KEY 环境变量未设置，调用评分会失败。")


# --------------------
# 5. 工具函数：根据 sentence_id 找到标准句
# --------------------
def find_sentence_by_id(sentence_id: str):
    for lesson in LESSON_DB.values():
        for s in lesson["sentences"]:
            if s["id"] == sentence_id:
                return s
    return None


# --------------------
# 6. 工具函数：用 Whisper 把音频转成西语文本
# --------------------
def transcribe_audio_to_text(audio_bytes: bytes) -> str:
    # 在临时文件里保存一下音频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # 调用 whisper 转写
    result = whisper_model.transcribe(tmp_path, language="es")
    text = result.get("text", "").strip()
    print("Whisper 转写结果：", text)
    return text


# --------------------
# 7. 工具函数：调用 DeepSeek 做评分
# --------------------
async def grade_with_deepseek(ref_text: str, user_text: str) -> dict:
    if not DEEPSEEK_API_KEY:
        # 这里直接抛错，前端会收到 500
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")

    system_prompt = (
        "你是一名严格但友好的西班牙语口语老师，负责给学生的朗读打分。\n"
        "请只用 JSON 格式回答，不要任何多余说明。\n\n"
        "JSON 字段包括：\n"
        "overall_score: 0-100 的整数，总分\n"
        "accuracy: '高' 或 '中' 或 '低'，发音准确度\n"
        "fluency: '高' 或 '中' 或 '低'，流利度\n"
        "integrity: '高' 或 '中' 或 '低'，是否读全\n"
        "missing_words: 漏读的单词数组\n"
        "mispronounced_words: 可能读错的单词数组\n"
        "suggestions: 三条简短的中文建议数组\n"
    )

    user_prompt = (
        f"【标准句】：{ref_text}\n"
        f"【学生朗读转写】：{user_text}\n\n"
        "请根据学生的朗读和标准句进行对比打分，输出上述 JSON。"
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # DeepSeek 返回结构里，内容在 choices[0].message.content 里，是一个 JSON 字符串
    content = data["choices"][0]["message"]["content"]
    import json

    result = json.loads(content)
    return result


# --------------------
# 8. /evaluate 接口：整合转写 + 打分
# --------------------
@app.post("/evaluate")
async def evaluate(
    file: UploadFile = File(...),
    sentence_id: str = Form(...)
):
    # 1. 找标准句
    ref_sentence = find_sentence_by_id(sentence_id)
    if not ref_sentence:
        raise HTTPException(status_code=404, detail="Sentence not found")

    ref_text = ref_sentence["text"]

    # 2. 读取音频
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # 3. Whisper 转写
    try:
        user_text = transcribe_audio_to_text(audio_bytes)
    except Exception as e:
        print("Whisper 转写出错：", e)
        raise HTTPException(status_code=500, detail="Transcription failed")

    # 4. DeepSeek 打分
    try:
        eval_result = await grade_with_deepseek(ref_text, user_text)
    except Exception as e:
        print("DeepSeek 评分出错：", e)
        raise HTTPException(status_code=500, detail="Grading failed")

    # 5. 把 sentence_id 带回去
    eval_result["sentence_id"] = sentence_id
    return eval_result


if __name__ == "__main__":
    # 本地调试用，Render 上可以用类似的启动命令
    uvicorn.run(app, host="0.0.0.0", port=8000)
