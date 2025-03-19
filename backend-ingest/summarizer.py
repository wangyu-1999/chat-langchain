import os
from config import PROMPT
from llm_service import model_manager
import json
import logging
import re
import asyncio

logger = logging.getLogger("backend")


class ChatModel:
    def __init__(self):
        self.client = model_manager.zhipu_client
        self.model = "glm-4-plus"
        # 控制并发请求数量
        self.semaphore = asyncio.Semaphore(3)  # 同时处理3个请求

    async def chat(self, messages):
        if messages is None or not str(messages).strip() or messages == "null":
            logger.warning("收到空消息、None 或 'null' 字符串")
            return {
                "title_cn": "",
                "title_en": "",
                "subject": "",
                "location": "",
                "summary": "",
                "english_summary": "",
            }

        async with self.semaphore:  # 使用信号量控制并发
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": f"{PROMPT}{messages}"}],
                    temperature=0.4,
                    top_p=0.4,
                    max_tokens=1024,
                )

                response_content = response.choices[0].message.content
                # 清理响应内容，移除可能的多余字符
                cleaned_content = response_content.strip()

                # 使用正则表达式匹配被代码块包裹的内容
                code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
                match = re.search(code_block_pattern, cleaned_content)

                if match:
                    # 如果找到匹配，提取第一个捕获组的内容
                    cleaned_content = match.group(1).strip()

                cleaned_content = "".join(
                    char for char in cleaned_content if ord(char) >= 32
                )

                logger.debug(f"cleaned_content: {cleaned_content}")

                try:
                    return json.loads(cleaned_content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析错误: {e}")
                    logger.error(f"原始消息: {messages}")
                    logger.error(f"API 响应: {response_content}")
                    return {
                        "title_cn": "",
                        "title_en": "",
                        "subject": "",
                        "location": "",
                        "summary": "",
                        "english_summary": "",
                    }
            except Exception as e:
                error_str = str(e)
                # 检查是否为内容过滤错误
                if '"code":"1301"' in error_str and "contentFilter" in error_str:
                    return {
                        "title_cn": "",
                        "title_en": "",
                        "subject": "",
                        "location": "",
                        "summary": "",
                        "english_summary": "",
                    }
                logger.error(f"API 调用错误: {e}")
                logger.error(f"原始消息: {messages}")
                return {
                    "title_cn": "",
                    "title_en": "",
                    "subject": "",
                    "location": "",
                    "summary": "",
                    "english_summary": "",
                }
