"""Main entrypoint for the app."""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from asyncio import Semaphore
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor

# 添加统一的日志配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("backend")

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

# 创建信号量实例，限制最大并发数为3
chat_semaphore = Semaphore(3)

# 添加摄入任务的信号量
ingest_semaphore = Semaphore(1)  # 限制只能同时运行一个摄入任务

# 添加ProcessPoolExecutor用于CPU密集型任务
process_pool = ProcessPoolExecutor(max_workers=1)

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
    async with chat_semaphore:  # 使用信号量控制并发
        response = await answer_chain.ainvoke(
            {"question": request.question, "chat_history": request.chat_history}
        )
        print(response)
        return response


@app.post("/api/ingest")
async def ingest(background_tasks: BackgroundTasks):
    # 如果已经在运行，直接返回状态信息
    if ingest_status["is_running"]:
        return {
            "status": "running",
            "message": "数据摄入任务正在执行中",
            "start_time": ingest_status["start_time"],
        }

    # 开始新的摄入任务
    ingest_status["is_running"] = True
    ingest_status["start_time"] = datetime.now().isoformat()
    ingest_status["last_error"] = None

    # 将耗时操作放入后台任务
    background_tasks.add_task(process_ingest)

    return {
        "status": "accepted",
        "message": "数据摄入任务已开始",
        "start_time": ingest_status["start_time"],
    }


async def process_ingest():
    """后台处理摄入任务"""
    try:
        # 直接调用 ingest_docs，不使用进程池
        final_stats = await ingest_docs()
        ingest_status["last_stats"] = final_stats
    except Exception as e:
        ingest_status["last_error"] = str(e)
        logger.error(f"Ingest error: {str(e)}")
    finally:
        ingest_status["is_running"] = False
        ingest_status["end_time"] = datetime.now().isoformat()


@app.get("/api/ingest/status")
async def get_ingest_status():
    return ingest_status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
