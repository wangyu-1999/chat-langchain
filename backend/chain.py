from dotenv import load_dotenv
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import weaviate
from config import (
    WEAVIATE_DOCS_INDEX_NAME,
    RESPONSE_TEMPLATE,
    REPHRASE_TEMPLATE,
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
import logging

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
        attributes=["source", "title", "date", "location", "subject"],
    )
    return vectorstore.as_retriever(search_kwargs=dict(k=3))


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()

    # 添加打印步骤
    def print_and_return(question: str):
        logging.info("\n=== 重写后的问题 ===\n%s", question)
        return question

    conversation_chain = (
        condense_question_chain | RunnableLambda(print_and_return)
    ) | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))),
            conversation_chain,
        ),
        (RunnableLambda(itemgetter("question")) | retriever),
    )


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}',source='{doc.metadata.get('source')}',date='{doc.metadata.get('date')}',location='{doc.metadata.get('location')}',subject='{doc.metadata.get('subject')}'>{doc.page_content}</doc>"
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

    # 添加调试信息来查看检索结果
    def debug_context(x):
        docs = x["docs"]
        logging.info("\n=== 向量检索结果 ===")
        for i, doc in enumerate(docs):
            logging.info("\n文档 %d:", i + 1)
            logging.info("内容: %s", doc.page_content)
            logging.info("来源: %s", doc.metadata.get("source", "unknown"))
        return format_docs(docs)

    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=debug_context)  # 替换原来的 lambda
        .with_config(run_name="RetrieveDocs")
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # 修改 debug_prompt 函数，确保所有必需的参数都被传入
    def debug_prompt(inputs):
        inputs = {**inputs, "number": 1}
        formatted_prompt = prompt.format_messages(**inputs)
        logging.info("\n=== 发送给模型的完整提示 ===")
        for msg in formatted_prompt:
            logging.info("\n%s:", msg.type)
            logging.info(msg.content)
        return formatted_prompt

    default_response_synthesizer = RunnableLambda(debug_prompt) | llm

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
