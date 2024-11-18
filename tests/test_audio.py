import asyncio
from cobwebAI_llmlib.audio import Audio2Text


def test_audio_pipeline():
    path = "assets/ai_lecture_3.m4a"
    a2t = Audio2Text(temp_dir="assets")
    text = asyncio.run(a2t.transcribe_file(path))
    with open(f"{path}.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text)
