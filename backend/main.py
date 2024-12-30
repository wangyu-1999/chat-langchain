"""Main entrypoint for the app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime

from chain import ChatRequest, answer_chain
from ingest import ingest_docs

# 获取项目根目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 使用简单的字典来存储状态
ingest_status = {
    "is_running": False,
    "last_error": None,
    "last_stats": None,
    "start_time": None,
    "end_time": None,
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = await answer_chain.ainvoke(
        {"question": request.question, "chat_history": request.chat_history}
    )
    print(response)
    return response


@app.post("/api/ingest")
async def ingest():
    if ingest_status["is_running"]:
        return {"status": "error", "message": "数据摄入任务已在运行中"}

    try:
        ingest_status["is_running"] = True
        ingest_status["last_error"] = None
        ingest_status["start_time"] = datetime.now().isoformat()

        final_stats = await ingest_docs()

        ingest_status["last_stats"] = final_stats
        ingest_status["end_time"] = datetime.now().isoformat()
        return {"status": "success", "message": "数据摄入完成", "stats": final_stats}

    except Exception as e:
        ingest_status["last_error"] = str(e)
        return {"status": "error", "message": str(e)}
    finally:
        ingest_status["is_running"] = False


@app.get("/api/ingest/status")
async def get_ingest_status():
    return ingest_status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
