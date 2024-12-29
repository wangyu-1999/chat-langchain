import os
from zhipuai import ZhipuAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.runnables import ConfigurableField
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

    @property
    def langchain_model(self):
        if not self._langchain_model:
            base_model = ChatZhipuAI(
                model="glm-4-flash",
                temperature=0,
                streaming=True,
                zhipuai_api_key=self.api_key,
            )
            self._langchain_model = base_model.configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="zhipu_glm_4",
            ).with_fallbacks([base_model])
        return self._langchain_model


model_manager = ModelManager()
