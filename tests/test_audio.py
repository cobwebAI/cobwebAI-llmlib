import asyncio
from cobwebai_llmlib.audio import Transcription


def test_audio_pipeline():
    path = "assets/ai_lecture_3.m4a"
    a2t = Transcription(_temp_dir="assets/temp")
    text = asyncio.run(a2t.transcribe_file(path))
    with open(f"{path}.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text)


if __name__ == "__main__":
    test_audio_pipeline()
