from openai import NOT_GIVEN, AsyncOpenAI
from asyncio import create_subprocess_exec
from aiofiles import ospath as aio_path
from langchain_playground.logger import log

AUDIO_TARGET_BITRATE = 128
"""Target bitrate for audio in Kb"""

OAI_AUDIO_LIMIT = 25 * 1024 * 8 * 1000 // AUDIO_TARGET_BITRATE
"""Maximum length of audio file for OpenAI API in milliseconds"""

AUDIO_SEGMENT_LENGTH = int(0.95 * OAI_AUDIO_LIMIT) // 1000
"""Seconds. Just to be safe, segments are 5% shorter than OAI_AUDIO_LIMIT"""

OAI_CLIENT = AsyncOpenAI()


async def preprocess_audio(path: str) -> list[str]:
    log.debug(f"preprocessing audio: {path}")
    
    ffmpeg = await create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        path,
        "-map",
        "0:a",
        "-c:a",
        "libopus",
        "-b:a",
        f"{AUDIO_TARGET_BITRATE}k",
        "-f",
        "segment",
        "-segment_time",
        f"{AUDIO_SEGMENT_LENGTH}",
        "-reset_timestamps",
        "1",
        f"{path}_%d.ogg",
    )

    exit_code = await ffmpeg.wait()

    if exit_code != 0:
        raise RuntimeError("audio segmentation: ffmpeg: nonzero status code")

    segment_paths: list[str] = []

    segment_idx = 0
    segment_path = f"{path}_{segment_idx}.ogg"

    while await aio_path.exists(segment_path):
        segment_paths.append(segment_path)

        segment_idx += 1
        segment_path = f"{path}_{segment_idx}.ogg"

    if len(segment_paths) == 0:
        raise RuntimeError("audio segmentation: no segments found")

    return segment_paths


async def audio_to_text(paths: list[str], language: str) -> list[str]:
    text_segments: list[str] = []

    for path in paths:
        # TODO: make this async somehow
        with open(path, "rb") as audio_file:
            prompt = text_segments[-1] if text_segments else NOT_GIVEN

            log.debug(f"transcribing: {path}")

            response = await OAI_CLIENT.audio.transcriptions.create(
                model="whisper-1", file=audio_file, language=language, prompt=prompt
            )

            text_segments.append(response.text)

    return text_segments


async def audio_pipeline(path: str, language: str = "ru") -> str | None:
    try:
        seg_paths = await preprocess_audio(path)
        text_segments = await audio_to_text(seg_paths, language=language)
        text = " ".join(text_segments)
        return text

    except Exception as e:
        log.error(f"audio pipeline failed with: {e}")
        return None
