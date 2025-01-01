import os
from config import PROMPT
from llm_service import model_manager
import json
import logging
import re

logger = logging.getLogger("backend")


class ChatModel:
    def __init__(self):
        self.client = model_manager.zhipu_client
        self.model = "glm-4-plus"

    def chat(self, messages):
        # 添加输入验证
        if messages is None or not isinstance(messages, str) or not messages.strip():
            logger.error(f"收到空消息或无效消息: {messages}")
            return {
                "title_cn": "",
                "title_en": "",
                "subject": "",
                "location": "",
                "summary": "",
                "english_summary": "",
            }

        try:
            response = self.client.chat.completions.create(
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

            print(cleaned_content)

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
