from config import PROMPT
from llm_service import model_manager
import json
import logging

logger = logging.getLogger("backend")


class ChatModel:
    def __init__(self):
        self.client = model_manager.zhipu_client
        self.model = "glm-4-plus"

    def chat(self, messages):
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

            # 改进 JSON 提取逻辑
            if "```json" in cleaned_content:
                # 找到 json 块的开始和结束
                start = cleaned_content.find("```json") + 7
                end = cleaned_content.rfind("```")
                if end > start:
                    cleaned_content = cleaned_content[start:end].strip()

            # 移除所有控制字符（包括换行符）
            cleaned_content = "".join(
                char for char in cleaned_content if ord(char) >= 32
            )

            try:
                # 尝试解析 JSON
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
                    "chinese_summary": "",
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
                "chinese_summary": "",
                "english_summary": "",
            }
