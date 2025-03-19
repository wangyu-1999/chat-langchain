import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()


class ModelManager:
    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self._zhipu_client = None
        self._langchain_model = None

    @property
    def zhipu_client(self):
        if not self._zhipu_client:
            self._zhipu_client = ZhipuAI(api_key=self.api_key)
        return self._zhipu_client


model_manager = ModelManager()
