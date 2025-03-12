import asyncio
from cobwebai_lib.text import TextPostProcessing, Test, Question
from uuid import UUID
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

client = AsyncAnthropic()


def test_text_fix():
    path = "assets/ai_lecture_3.m4a.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextPostProcessing(client)
    output_text = asyncio.run(
        processor.fix_transcribed_text(text, theme="Лекция по машинному обучению")
    )

    with open(f"{path}_fixed.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)


def test_conspect():
    path = "assets/ai_lecture_3.m4a.txt_fixed.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextPostProcessing(client)
    output_text = asyncio.run(
        processor.create_conspect_multi(
            [text], instructions="Лекция по машинному обучению"
        )
    )

    output_title = asyncio.run(processor.make_title(output_text))

    with open(f"{path}_conspect.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(f"OUTPUT_TITLE: {output_title} \n\n {output_text}")


def test_create_test():
    path = "assets/ai_lecture_3.m4a.txt_fixed.txt"
    text = open(path, encoding="utf-8").read()
    processor = TextPostProcessing(client)

    output: Test = asyncio.run(
        processor.make_test([text], "Не включай в тест вопросы про библиотеки.")
    )
    assert output != None

    print(output.test_name)
    for question in output.questions:
        print()
        print(question.question)
        print(f"T: {question.correct_answer}")
        print("\t" + question.correct_answer_explanation)
        for incorrect in question.incorrect_answers:
            print(f"X: {incorrect}")


if __name__ == "__main__":
    test_text_fix()
    test_conspect()
