from dotenv import load_dotenv
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import weaviate
from config import (
    WEAVIATE_DOCS_INDEX_NAME,
    RESPONSE_TEMPLATE,
    REPHRASE_TEMPLATE,
    RETRIEVER_TOP_K,
)
from ingest import get_embeddings_model
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from typing import Dict, List, Optional
from pydantic import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from llm_service import model_manager
from storage.azure_table import AzureTableStorage
from retriever_chain import create_retriever_chain

load_dotenv()


WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
    )
    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source"],
    )
    return vectorstore.as_retriever(search_kwargs=dict(k=RETRIEVER_TOP_K))


def format_docs(docs: Sequence[Document]) -> str:
    azure_storage = AzureTableStorage()
    formatted_docs = []

    for i, doc in enumerate(docs):
        # 从 Azure Table 获取完整内容，使用同步方式
        full_content = azure_storage.get_document_sync(doc.metadata["source"])
        if full_content:
            doc_string = f"<doc id='{i}',source='{doc.metadata['source']}',date='{full_content['date']}',location='{full_content['location']}',subject='{full_content['subject']}'>{full_content['english_summary']}</doc>"
            formatted_docs.append(doc_string)

    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    # 使用同步的 format_docs
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    default_response_synthesizer = prompt | llm

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="zhipu_glm_4",
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


retriever = get_retriever()
answer_chain = create_chain(model_manager.langchain_model, retriever)
