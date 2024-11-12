import asyncio
from langchain_playground.audio import audio_pipeline


def test_audio_pipeline():
    path = "assets/ai_lecture_3.m4a"

    text = asyncio.run(audio_pipeline(path))

    with open(f"{path}.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text)
