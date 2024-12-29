"""Main entrypoint for the app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from chain import ChatRequest, answer_chain

# 获取项目根目录的绝对路径
BASE_DIR = Path(__file__).resolve().parent.parent


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
