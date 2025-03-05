import asyncio
from cobwebai_lib.audio import Transcription


def test_audio_pipeline(
    temp_dir: str = "assets/temp",
    file_path: str = "assets/ai_lecture_3.m4a",
):
    a2t = Transcription(_temp_dir=temp_dir)
    text = asyncio.run(a2t.transcribe_file(file_path))
    with open(f"{file_path}.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(text)


if __name__ == "__main__":
    test_audio_pipeline()
