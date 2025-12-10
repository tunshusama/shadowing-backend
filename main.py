from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# 允许跨域（小程序需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), sentence_id: str = Form(...)):
    """
    假评分接口：不处理文件，只返回固定 JSON
    """
    return {
        "sentence_id": sentence_id,
        "overall_score": 88,
        "accuracy": "中",
        "fluency": "高",
        "integrity": "高",
        "missing_words": ["Ana"],
        "mispronounced_words": ["Hola"],
        "suggestions": [
            "整体不错，可以再放慢一些速度。",
            "注意句尾语调。",
            "多模仿原音的节奏。"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
