from typing import Generator, Tuple
from uuid import UUID, uuid5
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import AsyncHttpClient
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

    async def delete_embeddings(self, user_id: UUID, embedding_ids: list[UUID]) -> bool:
        """Removes embeddings from user's collection.

        Returns:
            bool: false if nothing changed.
        """
        user_id = str(user_id)

        try:
            chroma = await AsyncHttpClient(port=self.chroma_port)
            collection = await chroma.get_collection(
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
            chroma = await AsyncHttpClient(port=self.chroma_port)
            collection = await chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )

            await collection.delete(where={"project_id": project_id})

        except InvalidCollectionException:
            return False
        except ValueError:
            return False

        return True

    async def delete_document(self, user_id: UUID, document_id: UUID) -> bool:
        """Removes all data associated with the document from user's collection.

        Returns:
            bool: false if nothing changed.
        """
        user_id = str(user_id)
        document_id = str(document_id)

        try:
            chroma = await AsyncHttpClient(port=self.chroma_port)
            collection = await chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )

            await collection.delete(where={"document_id": document_id})

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
            uid = uuid5(UUID(int=0x12345678123456781234567812345678), part.page_content)
            content = part.page_content
            meta = part.metadata
            yield (uid, content, meta)

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

        user_id = str(user_id)
        project_id = str(project_id)

        where_meta = []

        if project_id:
            where_meta.append({"project_id": project_id})

        if document_id:
            where_meta.append({"document_id": document_id})

        try:
            chroma = await AsyncHttpClient(port=self.chroma_port)
            collection = await chroma.get_collection(
                user_id,
                embedding_function=self.embed_model,
            )

            retrieved = await collection.query(
                query_texts=query,
                where={"$and": where_meta} if where_meta else None,
                n_results=n_results,
                include=["documents"],
            )

            return retrieved["documents"]

        except InvalidCollectionException:
            return []
        except ValueError:
            return []

    async def add_document_to_project(
        self, user_id: UUID, project_id: UUID, document_id: UUID, text: str
    ) -> list[UUID]:
        """Splits and embeds text into Chroma.

        Returns:
            list[UUID]: ids of created embeddings.
        """

        user_id = str(user_id)
        project_id = str(project_id)

        chroma = await AsyncHttpClient(port=self.chroma_port)
        collection = await chroma.get_or_create_collection(
            user_id,
            embedding_function=self.embed_model,
        )

        documents = [
            Document(
                text, metadata={"project_id": project_id, "document_id": document_id}
            )
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

    async def store_and_retrieve(
        self,
        user_id: UUID,
        project_id: UUID,
        document_id: UUID,
        content: str,
        query: str,
        n_results: int = 1,
    ) -> list[str]:
        await self.add_document_to_project(user_id, project_id, document_id, content)
        return await self.retrieve(user_id, query, project_id, document_id, n_results)
