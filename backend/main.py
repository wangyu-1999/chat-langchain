"""Main entrypoint for the app."""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from chain import ChatRequest, answer_chain
from ingest import ingest_docs

# 获取项目根目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 添加一个全局变量来追踪摄入状态
ingest_status = {
    "is_running": False,
    "last_error": None,
    "last_stats": None,
    "start_time": None,
    "end_time": None,
}


# 创建异步摄入函数
async def run_ingest_task():
    global ingest_status
    from datetime import datetime

    try:
        ingest_status["is_running"] = True
        ingest_status["last_error"] = None
        ingest_status["start_time"] = datetime.now().isoformat()

        final_stats = await ingest_docs()  # 现在接收返回的统计信息

        ingest_status["last_stats"] = final_stats
        ingest_status["end_time"] = datetime.now().isoformat()
        ingest_status["is_running"] = False

    except Exception as e:
        ingest_status["last_error"] = str(e)
        ingest_status["is_running"] = False
        ingest_status["end_time"] = datetime.now().isoformat()
        raise e


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
async def ingest(background_tasks: BackgroundTasks):
    if ingest_status["is_running"]:
        return {"status": "error", "message": "数据摄入任务已在运行中"}

    background_tasks.add_task(run_ingest_task)
    return {"status": "success", "message": "数据摄入任务已开始"}


# 添加一个新端点来检查摄入状态
@app.get("/api/ingest/status")
async def get_ingest_status():
    return {
        "is_running": ingest_status["is_running"],
        "last_error": ingest_status["last_error"],
        "last_stats": ingest_status["last_stats"],
        "start_time": ingest_status["start_time"],
        "end_time": ingest_status["end_time"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
