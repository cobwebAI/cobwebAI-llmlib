from typing import Generator, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import AsyncHttpClient
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from uuid import UUID, uuid5
import asyncio


class VectorDB:
    """Provides interface to ChromaDB with users and projects.
    Each user is represented as a Chroma collection.
    Projects are distinguished by documents' metadata.
    Embeddings stored in a collection are guaranteed to be unique (by overwriting).
    """

    def __init__(
        self,
        embed_model_name: str,
        chroma_port: int,
        oai_key: str,
    ) -> None:
        """Constructs VectorDB instance.

        Args:
            embed_model_name (str): embedding function to use;
            chroma_port (int): ChromaDB server's http port;
            oai_key (str): OpenAI key (for embedding purposes).
        """

        self.chroma = asyncio.run(AsyncHttpClient(port=chroma_port))
        self.embed_model = OpenAIEmbeddingFunction(
            model_name=embed_model_name,
            api_key=oai_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    async def delete_embeddings(self, user_id: UUID, embedding_ids: list[UUID]) -> bool:
        """Removes embeddings from user's collection.

        Returns:
            bool: false if nothing changed.
        """
        user_id = str(user_id)

        try:
            collection = await self.chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )
        except InvalidCollectionException:
            return False
        except ValueError:
            return False

        await collection.delete(ids=list(map(str, embedding_ids)))
        return True

    async def delete_project(self, user_id: UUID, project_id: UUID) -> bool:
        """Removes all data associated with project from user's collection.

        Returns:
            bool: false if nothing changed.
        """
        user_id = str(user_id)
        project_id = str(project_id)

        try:
            collection = await self.chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )

            await collection.delete(where={"project_id": project_id})

        except InvalidCollectionException:
            return False
        except ValueError:
            return False

        return True

    async def delete_user(self, user_id: UUID) -> bool:
        """Removes user's collection from Chroma.

        Returns:
            bool: false if there was no such user-collection.
        """

        user_id = str(user_id)

        try:
            await self.chroma.get_collection(
                user_id, embedding_function=self.embed_model
            )
        except InvalidCollectionException:
            return False
        except ValueError:
            return False

        await self.chroma.delete_collection(user_id)
        return True

    def _split_documents(
        self, documents: list[Document]
    ) -> Generator[Tuple[UUID, str, dict], None, None]:
        for part in self.splitter.split_documents(documents):
            uid = uuid5(UUID(int=0x12345678123456781234567812345678), part.page_content)
            content = part.page_content
            meta = part.metadata
            yield (uid, content, meta)

    async def add_text_to_project(
        self, user_id: UUID, project_id: UUID, text: str
    ) -> list[UUID]:
        """Splits and embeds text into Chroma.

        Args:
            text (str): literally any string.

        Returns:
            list[UUID]: ids of created embeddings.
        """

        user_id = str(user_id)
        project_id = str(project_id)

        collection = await self.chroma.get_or_create_collection(
            user_id,
            embedding_function=self.embed_model,
        )

        documents = [
            Document(text, metadata={"user_id": user_id, "project_id": project_id})
        ]

        ids = []
        contents = []
        metas = []

        for uid, content, meta in self._split_documents(documents):
            ids.append(uid)
            contents.append(content)
            metas.append(meta)

        await collection.add(
            ids=list(map(str, ids)), metadatas=metas, documents=contents
        )

        return ids

    async def retrieve(
        self, user_id: UUID, project_id: UUID, query: str, n_results: int = 1
    ) -> list[str]:
        """Retrieves texts that are closest to query in embedding space.

        Args:
            query (str): search input.
            n_results (int, optional): number of neighbors to find. Defaults to 1.

        Returns:
            list[str]: nearest texts.
        """

        user_id = str(user_id)
        project_id = str(project_id)

        try:
            collection = await self.chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )

            retrieved = await collection.query(
                query_texts=query,
                where={"project_id": project_id},
                n_results=n_results,
                include=["documents"],
            )

            return retrieved["documents"]

        except InvalidCollectionException:
            return []
        except ValueError:
            return []
