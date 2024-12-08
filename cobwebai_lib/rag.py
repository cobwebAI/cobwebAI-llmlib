from logging import Logger
from typing import Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import AsyncHttpClient
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from .logger import log as local_log
import asyncio


class RAG:
    def __init__(
        self,
        chroma_port: int = 35432,
        log: Logger | None = None,
        **kwargs,
    ) -> None:
        self.model_name = "gpt-4o-mini"
        self.embedding_model_name = "text-embedding-ada-002"
        self.llm = ChatOpenAI(model=self.model_name)

        self.chroma_port = chroma_port
        self.log = log if log else local_log
        self.chroma = asyncio.run(AsyncHttpClient(port=self.chroma_port))

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        self.chroma_embeddings = OpenAIEmbeddingFunction(
            model_name=self.embedding_model_name,
            api_key=self.llm.openai_api_key.get_secret_value(),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    async def clear_project(self, user_id: str, project_id: str):
        collection = await self.chroma.get_or_create_collection(
            name=f"{user_id}",
            embedding_function=self.chroma_embeddings,
        )

        await collection.delete(where={"project_id": project_id})
        
    def split_documents(self, documents: list[Document]) -> Tuple[list[str], list[str], list[dict]]:
        splits = self.splitter.split_documents(documents)

        ids: list[str] = []
        parts: list[str] = []
        metadatas: list[dict] = []

        for part in splits:
            ids.append(str(hash(part.page_content)))
            parts.append(part.page_content)
            metadatas.append(part.metadata)
            
        return (ids, parts, metadatas)

    async def add_note_to_project(self, user_id: str, project_id: str, note: str):
        collection = await self.chroma.get_or_create_collection(
            name=f"{user_id}",
            embedding_function=self.chroma_embeddings,
        )

        ids, documents, metadatas = self.split_documents([
            Document(note, metadata={"user_id": user_id, "project_id": project_id})
        ])

        await collection.add(ids=ids, metadatas=metadatas, documents=documents)

    async def retrieve(self, user_id: str, project_id: str, query: str) -> list[str]:
        collection = await self.chroma.get_or_create_collection(
            name=f"{user_id}",
            embedding_function=self.chroma_embeddings,
        )

        retrieved = await collection.query(
            query_texts=query, where={"project_id": project_id}, n_results=2
        )

        return retrieved
