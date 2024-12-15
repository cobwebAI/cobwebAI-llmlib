from typing import Generator, Tuple
from loguru import logger
from uuid import UUID, uuid5
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import AsyncHttpClient
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api import AsyncClientAPI
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)


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

        self.log = logger
        self.chroma_port = chroma_port
        self.embed_model = OpenAIEmbeddingFunction(
            model_name=embed_model_name,
            api_key=oai_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

    async def _get_or_create_chroma_collection(
        self, user_id: UUID
    ) -> tuple[AsyncClientAPI, AsyncCollection]:
        chroma = await AsyncHttpClient(port=self.chroma_port)
        collection = await chroma.get_or_create_collection(
            user_id,
            embedding_function=self.embed_model,
        )

        return (chroma, collection)

    async def _try_get_chroma_collection(
        self, user_id: UUID
    ) -> tuple[AsyncClientAPI, AsyncCollection] | tuple[None, None]:
        try:
            chroma = await AsyncHttpClient(port=self.chroma_port)
            collection = await chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )
            return (chroma, collection)
        except InvalidCollectionException:
            return (None, None)
        except ValueError:
            return (None, None)

    async def delete_embeddings(self, user_id: UUID, embedding_ids: list[UUID]) -> bool:
        """Removes embeddings from user's collection.

        Returns:
            bool: False if no such user.
        """

        chroma, collection = await self._try_get_chroma_collection(user_id)

        if collection is None:
            return False

        await collection.delete(ids=list(map(str, embedding_ids)))

        return True

    async def delete_project(self, user_id: UUID, project_id: UUID) -> bool:
        """Removes all data associated with project from user's collection.

        Returns:
            bool: False if no such user.
        """
        project_id = str(project_id)

        chroma, collection = await self._try_get_chroma_collection(user_id)

        if collection is None:
            return False

        await collection.delete(where={"project_id": project_id})

        return True

    async def invalidate_document(self, user_id: UUID, document_id: UUID) -> bool:
        """Removes all data associated with the document from user's collection.

        Returns:
            bool: False if no such user.
        """
        document_id = str(document_id)

        chroma, collection = await self._try_get_chroma_collection(user_id)

        if collection is None:
            return False

        await collection.delete(where={"document_id": document_id})

        return True

    async def delete_user(self, user_id: UUID) -> bool:
        """Removes user's collection from Chroma.

        Returns:
            bool: False if there was no such user-collection.
        """

        user_id = str(user_id)

        try:
            chroma = await AsyncHttpClient(port=self.chroma_port)
            await chroma.get_collection(user_id, embedding_function=self.embed_model)
        except InvalidCollectionException:
            return False
        except ValueError:
            return False

        await chroma.delete_collection(user_id)
        return True

    def _split_documents(
        self, documents: list[Document]
    ) -> Generator[Tuple[UUID, str, dict], None, None]:
        for part in self.splitter.split_documents(documents):
            content = part.page_content
            meta = part.metadata

            to_hash = content + meta["project_id"] + meta["document_id"]
            uuid_from_hash = uuid5(
                UUID(int=0x12345678123456781234567812345678), to_hash
            )

            yield (uuid_from_hash, content, meta)

    def _prepare_document(
        self, project_id: UUID, document_id: UUID, text: str
    ) -> tuple[list[UUID], list[str], list[str], list[dict]]:
        document = Document(
            text,
            metadata={"project_id": str(project_id), "document_id": str(document_id)},
        )

        ids, str_ids, contents, metas = [], [], [], []

        for uid, content, meta in self._split_documents([document]):
            ids.append(uid)
            str_ids.append(str(uid))
            contents.append(content)
            metas.append(meta)

        return (ids, str_ids, contents, metas)

    async def retrieve(
        self,
        user_id: UUID,
        query: str,
        project_id: UUID | None = None,
        document_id: UUID | None = None,
        n_results: int = 1,
    ) -> list[str]:
        """Retrieves texts that are closest to query in embedding space.
        Can query by project AND/OR by document id.

        Args:
            query (str): search input.
            n_results (int, optional): number of neighbors to find. Defaults to 1.

        Returns:
            list[str]: nearest texts.
            Empty on exception or if BOTH project and query ids are None.
        """

        if not project_id and not document_id:
            return []

        where_meta = []

        if project_id:
            where_meta.append({"project_id": str(project_id)})

        if document_id:
            where_meta.append({"document_id": str(document_id)})

        chroma, collection = await self._try_get_chroma_collection(user_id)

        if collection is None:
            self.log.warning(f"Collection for user `{user_id}` doesn't exist")
            return []

        retrieved = await collection.query(
            query_texts=query,
            where={"$and": where_meta} if where_meta else None,
            n_results=n_results,
            include=["documents"],
        )

        return retrieved["documents"][0]

    async def add_document_to_project(
        self, user_id: UUID, project_id: UUID, document_id: UUID, text: str
    ) -> list[UUID]:
        """Splits and embeds text into Chroma.

        Returns:
            list[UUID]: ids of created embeddings.
        """

        chroma, collection = await self._get_or_create_chroma_collection(user_id)

        ids, str_ids, contents, metas = self._prepare_document(
            project_id, document_id, text
        )

        await collection.add(ids=str_ids, metadatas=metas, documents=contents)

        return ids

    async def add_document_and_query(
        self,
        user_id: UUID,
        project_id: UUID,
        document_id: UUID,
        text: str,
        query: str,
        n_results: int = 2,
    ) -> list[str]:
        """Adds document to chroma with overwriting and then queries it.

        Args:
            query (str): search input.
            n_results (int, optional): number of neighbors to find. Defaults to 1.

        Returns:
            list[str]: texts of embeddings closest to query.
        """

        chroma, collection = await self._get_or_create_chroma_collection(user_id)

        ids, str_ids, contents, metas = self._prepare_document(
            project_id, document_id, text
        )

        await collection.add(ids=str_ids, metadatas=metas, documents=contents)

        retrieved = await collection.query(
            query_texts=query,
            where={"document_id": str(document_id)},
            n_results=n_results,
            include=["documents"],
        )

        return retrieved["documents"][0]

    async def store_and_retrieve(
        self,
        user_id: UUID,
        project_id: UUID,
        document_id: UUID,
        content: str,
        query: str,
        n_results: int = 2,
    ) -> list[str]:
        """Optimized function to use with chatbots.
        Queries document with or without storing it first.

        Note: document must be invalidated properly on change!

        Args:
            content (str): document's content.
            query (str): search input.
            n_results (int, optional): number of neighbors to find. Defaults to 1.

        Returns:
            list[str]: texts of embeddings closest to query.
            Might be empty.
        """

        content = content.strip()
        query = query.strip()
        str_document_id = str(document_id)

        chroma, collection = await self._try_get_chroma_collection(user_id)

        if collection is not None:
            metadatas = await collection.get(
                where={"document_id": str_document_id}, include=["metadatas"]
            )

            if metadatas["ids"]:
                retrieved = await collection.query(
                    query_texts=query,
                    where={"document_id": str_document_id},
                    n_results=n_results,
                    include=["documents"],
                )

                return retrieved["documents"][0]

        del collection
        del chroma

        return await self.add_document_and_query(
            user_id, project_id, document_id, content, query, n_results
        )
