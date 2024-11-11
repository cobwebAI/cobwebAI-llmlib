from openai import NOT_GIVEN, OpenAI
from pydub import AudioSegment

AUDIO_TARGET_BITRATE = 256

OAI_AUDIO_LIMIT = 25 * 1024 * 8 * 1000 // AUDIO_TARGET_BITRATE
"""Maximum length of audio file for OpenAI API in milliseconds"""


def preprocess_audio(path: str) -> list[str]:
    audio: AudioSegment = AudioSegment.from_file(path)

    duration = int(audio.duration_seconds * 1000)
    full_seg_cnt = duration // OAI_AUDIO_LIMIT
    last_seg_dur = duration % OAI_AUDIO_LIMIT
    seg_cnt = full_seg_cnt if last_seg_dur == 0 else full_seg_cnt + 1

    segments: list[AudioSegment] = []

    for i in range(seg_cnt):
        segments.append(
            audio[i * OAI_AUDIO_LIMIT : min((i + 1) * OAI_AUDIO_LIMIT, duration)]
        )

    segment_paths: list[str] = []

    for i, seg in enumerate(segments):
        seg_path = f"{path}.{i}.mp3"
        seg.export(seg_path, format="mp3", bitrate=f"{AUDIO_TARGET_BITRATE}K")
        segment_paths.append(seg_path)

    return segment_paths


def audio_to_text(paths: list[str], language: str = "ru") -> list[str]:
    oai_client = OpenAI()
    texts: list[str] = []

    with open("assets/oai.log", "w", encoding="utf-8") as log:
        for path in paths:
            with open(path, "rb") as audio_file:
                prompt = texts[-1] if texts else NOT_GIVEN
                response = oai_client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, language=language, prompt=prompt
                )
                log.write(response.to_json() + "\n\n")
                texts.append(response.text)

    return texts


def pipeline():
    seg_paths = preprocess_audio("assets/ai_lecture_3.m4a")
    texts = audio_to_text(seg_paths)

    txt_paths = [f"{path}.txt" for path in seg_paths]

    for path, text in zip(txt_paths, texts):
        with open(path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)

    with open("assets/ai_lecture_3.txt", "w", encoding="utf-8") as txt_file:
        for path in txt_paths:
            with open(path, "r", encoding="utf-8") as in_txt_file:
                txt_file.write(in_txt_file.read() + " ")


if __name__ == "__main__":
    pipeline()
