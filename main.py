import os
import json
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

app = FastAPI()

# 允许所有来源访问，方便小程序调试和线上调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============== 课程数据（写死一节课，和前端保持一致） ===============
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


def find_sentence_by_id(sentence_id: str):
    for lesson in LESSON_DB.values():
        for s in lesson["sentences"]:
            if s["id"] == sentence_id:
                return s
    return None


# =============== 读取环境变量：DeepSeek 和 AssemblyAI ===============
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


# =============== 调用 AssemblyAI：上传音频 + 转写（西语） ===============
async def transcribe_with_assemblyai(audio_bytes: bytes) -> str:
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError("ASSEMBLYAI_API_KEY 未配置")

    headers_upload = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/octet-stream",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # 1. 上传音频
        upload_resp = await client.post(
            f"{ASSEMBLYAI_BASE_URL}/upload",
            headers=headers_upload,
            content=audio_bytes
        )
        upload_resp.raise_for_status()
        upload_url = upload_resp.json()["upload_url"]

        # 2. 创建转写任务（指定西班牙语）
        headers_json = {
            "authorization": ASSEMBLYAI_API_KEY,
            "content-type": "application/json",
        }
        transcript_payload = {
            "audio_url": upload_url,
            "language_code": "es",
            "punctuate": True
        }
        create_resp = await client.post(
            f"{ASSEMBLYAI_BASE_URL}/transcript",
            headers=headers_json,
            json=transcript_payload
        )
        create_resp.raise_for_status()
        transcript_id = create_resp.json()["id"]

        # 3. 轮询任务状态，直到完成或超时
        status_url = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"

        for _ in range(30):  # 最多等 30 秒
            status_resp = await client.get(status_url, headers=headers_json)
            status_resp.raise_for_status()
            data = status_resp.json()
            status = data.get("status")
            if status == "completed":
                text = data.get("text", "").strip()
                print("AssemblyAI 转写结果：", text)
                return text
            elif status == "error":
                print("AssemblyAI 转写出错：", data.get("error"))
                raise RuntimeError("Transcription error from AssemblyAI")

            await asyncio.sleep(1)

        raise RuntimeError("Transcription timeout")


# =============== 调用 DeepSeek：根据标准句 + 用户句打分 ===============
async def grade_with_deepseek(ref_text: str, user_text: str) -> dict:
    if not DEEPSEEK_API_KEY:
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

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    result = json.loads(content)
    return result


# =============== evaluate：整合 ASR + DeepSeek 的主接口 ===============
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

    # 3. 调用 AssemblyAI 转写
    try:
        user_text = await transcribe_with_assemblyai(audio_bytes)
    except Exception as e:
        print("AssemblyAI 转写出错：", e)
        raise HTTPException(status_code=500, detail="Transcription failed")

    # 4. 调用 DeepSeek 打分
    try:
        eval_result = await grade_with_deepseek(ref_text, user_text)
    except Exception as e:
        print("DeepSeek 评分出错：", e)
        raise HTTPException(status_code=500, detail="Grading failed")

    eval_result["sentence_id"] = sentence_id
    eval_result["user_text"] = user_text
    return eval_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
