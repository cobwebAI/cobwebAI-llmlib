import asyncio
from cobwebai_llmlib.text import TextProcessing


def test_text_fix():
    path = "assets/ai_lecture_3.m4a.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextProcessing()
    output_text = asyncio.run(
        processor.fix_transcribed_text(text, theme="Лекция по машинному обучению.")
    )

    with open(f"{path}_fixed.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)
