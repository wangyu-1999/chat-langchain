import os
from prompts.get_news_template import PROMPT
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import json
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class ChatModel:
    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        self.model = "glm-4-plus"  # 使用基础GLM-4模型

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

            # 如果响应内容被包裹在 ```json ``` 中，提取其中的 JSON 内容
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.replace("```json", "", 1)
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                cleaned_content = cleaned_content.strip()

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
