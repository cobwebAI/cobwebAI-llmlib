from openai import NOT_GIVEN, AsyncOpenAI
import assemblyai as aai
from asyncio import create_subprocess_exec
from aiofiles import ospath as aio_path
from loguru import logger
from aiofiles import tempfile, open as aio_open
from uuid import uuid4
import asyncio
import os


class Transcription:
    """Converts audio/video files to text.
    Might perform chunking of input files."""

    AUDIO_TARGET_BITRATE = 128
    """Target bitrate for audio in Kb"""

    OAI_AUDIO_LIMIT = 25 * 1024 * 8 * 1000 // AUDIO_TARGET_BITRATE
    """Maximum length of audio file for OpenAI API in milliseconds"""

    AUDIO_SEGMENT_LENGTH = int(0.85 * OAI_AUDIO_LIMIT) // 1000
    """Seconds. Just to be safe, segments are 5% shorter than OAI_AUDIO_LIMIT"""

    AUDIO_SEGMENT_LENGTH_AAI = 2 * 60 * 60
    """Seconds. Arbitrary max duration for AssemblyAI API"""

    def __init__(
        self,
        client: AsyncOpenAI | aai.Client | None = None,
        force_oai: bool = False,
        _temp_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Constructs transcriber with provided resources or its own"""

        self.temp_dir = _temp_dir
        self.log = logger

        self.using_aai = False
        self.chunk_duration = self.AUDIO_SEGMENT_LENGTH

        if client:
            self.client = client
        elif not force_oai:
            self.client = aai.Client(settings=aai.settings)
        else:
            self.client = AsyncOpenAI(**kwargs)

        if isinstance(self.client, aai.Client):
            self.transcriber = aai.Transcriber(client=self.client)
            self.using_aai = True
            self.chunk_duration = self.AUDIO_SEGMENT_LENGTH_AAI

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
            f"{self.chunk_duration}",
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

        if self.using_aai:
            for i, path in enumerate(paths):
                self.log.debug(f"transcribing group: part {i}: `{path}`")

            ts_config = aai.TranscriptionConfig(
                language_code=language, format_text=True
            )

            transcripts: aai.TranscriptGroup = await asyncio.wrap_future(
                self.transcriber.transcribe_group_async(paths, ts_config)
            )

            if transcripts.status == aai.TranscriptStatus.error:
                raise RuntimeError(f"audio transcription: {transcripts.error}")

            text_segments.extend(map(lambda t: t.text, transcripts.transcripts))

        else:
            for i, full_path in enumerate(paths):
                name = path.basename(full_path)
                async with aio_open(full_path, "rb") as audio_file:
                    prompt = text_segments[i - 1] if i > 0 else NOT_GIVEN

                    self.log.debug(f"transcribing: part {i}: `{full_path}`")

                    response = await self.oai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=(name, await audio_file.read()),
                        language=language,
                        prompt=prompt,
                    )

                    text_segments.append(response.text)

        return text_segments

    async def transcribe_file(self, path: str, language: str = "ru") -> str:
        """Transcribes input video/audio file into text"""

        async with tempfile.TemporaryDirectory() as tmpdir:
            seg_paths = await self.audio_segmentation(path, tmpdir)
            segments = await self.transcribe_segments(seg_paths, language=language)
            return self.stitch_text_segments(segments)
