from cobwebai_lib import LLMTools
from cobwebai_lib.chat import ChatAttachment
from asyncio import run
from os import environ
from uuid import UUID


async def chat():
    llm = LLMTools(environ["OPENAI_API_KEY"], 35432)

    user = UUID(int=0x12345678123456781234567812345675)
    project = user
    document = ChatAttachment(
        UUID(int=0x12345678123456781234567812345674),
        open("assets/ai_lecture_3_fixed_chunk3072.txt", encoding="utf-8").read(),
    )

    user_msg, bot_msg = await llm.chat_with_rag(
        user,
        project,
        "Что такое AlexNet?",
        [document],
    )

    print()
    print(user_msg)
    print()
    print(bot_msg)


def test_chat():
    run(chat())


if __name__ == "__main__":
    test_chat()
