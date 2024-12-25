"""Main entrypoint for the app."""
import asyncio
from typing import Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chain import ChatRequest, answer_chain

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
    print(request)
    """处理聊天请求"""
    response = await answer_chain.ainvoke(
        {
            "question": request.question,
            "chat_history": request.chat_history
        }
    )
    print(response)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
