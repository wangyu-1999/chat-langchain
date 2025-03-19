"""Main entrypoint for the app."""

from fastapi import FastAPI, BackgroundTasks, Security, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime
from asyncio import Semaphore
import logging
import os
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("backend")

load_dotenv()

from ingest import ingest_docs
from cluster_analysis import analyze_news_clusters
from create_schema import create_schema_if_not_exists

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

# 删除单独的ingest_semaphore，改用共享信号量
shared_task_semaphore = Semaphore(1)  # 限制只能同时运行一个任务（ingest或cluster）

# 添加聚类分析状态
cluster_analysis_status = {
    "is_running": False,
    "last_error": None,
    "last_result": None,
    "start_time": None,
    "end_time": None,
}

# 在启动服务之前创建 collection
create_schema_if_not_exists()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 添加 API key 配置
API_KEY = os.getenv("API_KEY")  # 建议使用环境变量
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header or api_key_header != API_KEY:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="无效的 API key")
    return api_key_header


@app.post("/api/ingest")
async def ingest(
    background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)
):
    # 检查是否有任务在运行
    if ingest_status["is_running"] or cluster_analysis_status["is_running"]:
        return {
            "status": "busy",
            "message": "系统正在处理其他任务，请稍后再试",
            "ingest_running": ingest_status["is_running"],
            "cluster_running": cluster_analysis_status["is_running"],
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
async def get_ingest_status(api_key: str = Depends(get_api_key)):
    return ingest_status


@app.post("/api/clusters")
async def analyze_clusters(
    background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)
):
    # 检查是否有任务在运行
    if ingest_status["is_running"] or cluster_analysis_status["is_running"]:
        return {
            "status": "busy",
            "message": "系统正在处理其他任务，请稍后再试",
            "ingest_running": ingest_status["is_running"],
            "cluster_running": cluster_analysis_status["is_running"],
        }

    # 开始新的分析任务
    cluster_analysis_status["is_running"] = True
    cluster_analysis_status["start_time"] = datetime.now().isoformat()
    cluster_analysis_status["last_error"] = None

    # 将耗时操作放入后台任务
    background_tasks.add_task(process_cluster_analysis)

    return {
        "status": "accepted",
        "message": "新闻聚类分析任务已开始",
        "start_time": cluster_analysis_status["start_time"],
    }


async def process_cluster_analysis():
    """后台处理聚类分析任务"""
    try:
        result = await analyze_news_clusters()
        cluster_analysis_status["last_result"] = result
    except Exception as e:
        cluster_analysis_status["last_error"] = str(e)
        logger.error(f"Cluster analysis error: {str(e)}")
    finally:
        cluster_analysis_status["is_running"] = False
        cluster_analysis_status["end_time"] = datetime.now().isoformat()


@app.get("/api/clusters/status")
async def get_cluster_analysis_status(api_key: str = Depends(get_api_key)):
    return cluster_analysis_status


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
