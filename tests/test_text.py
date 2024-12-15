import asyncio
from cobwebai_lib.text import TextPostProcessing


def test_text_fix():
    path = "assets/ai_lecture_3.m4a.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextPostProcessing()
    output_text = asyncio.run(
        processor.fix_transcribed_text(text, theme="Лекция по машинному обучению")
    )

    with open(f"{path}_fixed.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)


def test_conspect():
    path = "assets/ai_lecture_3_fixed_chunk3072.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextPostProcessing()
    output_text = asyncio.run(
        processor.create_conspect(text, theme="Лекция по машинному обучению")
    )

    with open(f"{path}_conspect.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)

if __name__ == "__main__":
    test_text_fix()
    test_conspect()
