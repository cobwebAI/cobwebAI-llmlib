from cobwebai_lib.rag import RAG
import asyncio

def test_rag_init():
    rag = RAG()
    text = open("assets/ai_lecture_3_fixed_chunk3072.txt", encoding='utf-8').read()
    
    asyncio.run(rag.clear_project("user_test", "project_test"))
    asyncio.run(rag.add_note_to_project("user_test", "project_test", text))
    retrieved = asyncio.run(rag.retrieve("user_test", "project_test", "MAE и MSE"))
    
    print(retrieved)


if __name__ == "__main__":
    test_rag_init()

