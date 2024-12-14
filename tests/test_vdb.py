from cobwebai_lib.vdb import VectorDB
from asyncio import run
from os import environ
from uuid import UUID


async def vdb():
    vdb = VectorDB("text-embedding-ada-002", 35432, environ["OPENAI_API_KEY"])
    text = open("assets/ai_lecture_3_fixed_chunk3072.txt", encoding="utf-8").read()
    user = UUID(int=0x12345678123456781234567812345678)
    project = user

    await vdb.delete_project(user, project)

    assert await vdb.add_document_to_project(
        user, project, text
    ) == await vdb.add_document_to_project(user, project, text)

    retrieved = await vdb.retrieve(user, project, "MAE и MSE", n_results=2)
    assert len(retrieved) > 0
    print(retrieved)

    assert await vdb.delete_project(user, project)
    assert await vdb.delete_user(user)

    embedding_ids = await vdb.add_document_to_project(user, project, text)
    assert await vdb.delete_embeddings(user, embedding_ids)


def test_vdb_init():
    run(vdb())


if __name__ == "__main__":
    test_vdb_init()
