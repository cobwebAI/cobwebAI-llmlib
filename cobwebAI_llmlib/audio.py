from logging import Logger
from openai import NOT_GIVEN, AsyncOpenAI
from asyncio import create_subprocess_exec
from aiofiles import ospath as aio_path
from cobwebai_llmlib import logger
from tempfile import TemporaryDirectory
from uuid import uuid4
import os


class Transcription:
    """Converts audio/video files to text.
    Internally splits audio into segments.
    """

    AUDIO_TARGET_BITRATE = 128
    """Target bitrate for audio in Kb"""

    OAI_AUDIO_LIMIT = 25 * 1024 * 8 * 1000 // AUDIO_TARGET_BITRATE
    """Maximum length of audio file for OpenAI API in milliseconds"""

    AUDIO_SEGMENT_LENGTH = int(0.95 * OAI_AUDIO_LIMIT) // 1000
    """Seconds. Just to be safe, segments are 5% shorter than OAI_AUDIO_LIMIT"""

    def __init__(
        self,
        oai_client: AsyncOpenAI | None = None,
        log: Logger | None = None,
        _temp_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Constructs transcriber with provided resources or it's own"""

        self.temp_dir = _temp_dir
        self.log = log if log else logger.log
        self.oai_client = oai_client if oai_client else AsyncOpenAI(**kwargs)

    @staticmethod
    async def find_segments(path_prefix: str) -> list[str]:
        """Finds audio segments ffmpeg created"""

        segments: list[str] = []
        segment_idx = 0
        segment_path = f"{path_prefix}_{segment_idx}.ogg"

        while await aio_path.exists(segment_path):
            segments.append(segment_path)
            segment_idx += 1
            segment_path = f"{path_prefix}_{segment_idx}.ogg"

        return segments

    @staticmethod
    def stitch_text_segments(segments: list[str]) -> str:
        """Combines text segments into final text"""
        # TODO: Improve heuristic
        return " ".join(segments)

    async def audio_segmentation(self, path: str, tmpdir: str) -> list[str]:
        """Splits input audio/video file into segments, returns paths"""

        self.log.debug(f"preprocessing audio: {path} in {tmpdir}")

        path_prefix = os.path.join(tmpdir, uuid4().hex)

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
            f"{self.AUDIO_TARGET_BITRATE}k",
            "-f",
            "segment",
            "-segment_time",
            f"{self.AUDIO_SEGMENT_LENGTH}",
            "-reset_timestamps",
            "1",
            f"{path_prefix}_%d.ogg",
        )

        exit_code = await ffmpeg.wait()
        segment_paths = await self.find_segments(path_prefix)

        if exit_code != 0:
            raise RuntimeError("audio segmentation: ffmpeg: nonzero status code")
        elif len(segment_paths) == 0:
            raise RuntimeError("audio segmentation: no segments created")

        return segment_paths

    async def transcribe_segments(self, paths: list[str], language: str) -> list[str]:
        """Transcribes a list of sequential audio segments into text segments"""
        text_segments: list[str] = []

        for path in paths:
            # TODO: make this async somehow
            with open(path, "rb") as audio_file:
                prompt = text_segments[-1] if text_segments else NOT_GIVEN

                self.log.debug(f"transcribing: {path}")

                response = await self.oai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    prompt=prompt,
                )

                text_segments.append(response.text)

        return text_segments

    async def transcribe_file(self, path: str, language: str = "ru") -> str | None:
        """Transcribes input video/audio file into text"""
        output: str | None = None

        # ! This is synchronous
        with TemporaryDirectory(self.temp_dir) as tmpdir:
            try:
                seg_paths = await self.audio_segmentation(path, tmpdir)
                segments = await self.transcribe_segments(seg_paths, language=language)
                output = self.stitch_text_segments(segments)

            except Exception as e:
                self.log.error(f"transcribe_file failed with: {e}")

        return output
