from cobwebai_lib.vdb import VectorDB
from asyncio import run
from os import environ
from uuid import UUID


def test_vdb_init():
    vdb = VectorDB("text-embedding-ada-002", 35432, environ["OPENAI_API_KEY"])
    text = open("assets/ai_lecture_3_fixed_chunk3072.txt", encoding="utf-8").read()
    user = UUID(int=0x12345678123456781234567812345678)
    project = user

    run(vdb.delete_project(user, project))

    assert run(vdb.add_text_to_project(user, project, text)) == run(
        vdb.add_text_to_project(user, project, text)
    )

    retrieved = run(vdb.retrieve(user, project, "MAE и MSE", n_results=2))
    
    assert len(retrieved) > 0

    print(retrieved)

    assert run(vdb.delete_project(user, project))
    assert run(vdb.delete_user(user))
    
    embedding_ids = run(vdb.add_text_to_project(user, project, text))
    assert run(vdb.delete_embeddings(user, embedding_ids))


if __name__ == "__main__":
    test_vdb_init()
