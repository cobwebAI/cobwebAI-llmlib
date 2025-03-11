from loguru import logger
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from semchunk import chunkerify
from pydantic import BaseModel
from .models import OAIModel, AnthModel, Message, Role
from .prompts import *


class Question(BaseModel):
    question: str
    correct_answer: str
    incorrect_answers: list[str]
    correct_answer_explanation: str


class Test(BaseModel):
    test_name: str
    questions: list[Question]


class TextPostProcessing:
    """Corrects speech recognition mistakes in a text chunk by chunk"""

    CHUNK_SIZE = 3072
    """Text will be split in chunks each this number of tokens at max"""

    CHUNKERIFY_MODEL_HINT = "gpt-4o-mini"

    def __init__(
        self,
        client: AsyncOpenAI | AsyncAnthropic | None = None,
        **kwargs,
    ) -> None:
        """Constructs text post-processor with provided resources or its own"""

        self.log = logger

        if isinstance(client, AsyncAnthropic):
            self.model = AnthModel(client, **kwargs)
        elif isinstance(client, AsyncOpenAI):
            self.model = OAIModel(client, **kwargs)
        else:
            self.model = AnthModel(ant_client=None, **kwargs)

        # Shoud I disable memoization here?
        self.splitter = chunkerify(
            self.CHUNKERIFY_MODEL_HINT, chunk_size=self.CHUNK_SIZE, memoize=False
        )

    def _chunking_system_prompt(
        self,
        prev_chunk: str | None = None,
        theme: str | None = None,
        from_text: bool = False,
    ) -> str:
        """Instantiates system prompt template with supplied info"""

        return "\n".join(
            [
                TEXT_FIX_PROMPT if from_text else AUDIO_FIX_PROMPT,
                THEME_PROMPT.format(theme=theme) if theme else "",
                PREV_CHUNK_PROMPT.format(chunk=prev_chunk) if prev_chunk else "",
            ]
        )

    async def fix_transcribed_text(
        self, text: str, theme: str | None = None, from_text: bool = False
    ) -> str:
        """Post-processes supplied text"""

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")

        if not text.strip():
            raise ValueError("no text supplied")

        if theme:
            theme = theme.strip()

        input_chunks = self.splitter(text)
        output_chunks: list[str] = []

        for i, current_chunk in enumerate(input_chunks):
            prev_chunk = output_chunks[i - 1] if i > 0 else None

            system_prompt = self._chunking_system_prompt(
                prev_chunk=prev_chunk,
                theme=(theme if theme else None),
                from_text=from_text,
            )

            messages = [
                Message(Role.SYSTEM, system_prompt),
                Message(Role.USER, current_chunk),
            ]

            output_chunk = await self.model.invoke_simple(messages)
            output_chunks.append(output_chunk)

        return " ".join(output_chunks)

    async def create_conspect_multi(
        self, texts: list[str], instructions: str | None = None
    ) -> str:
        """Creates an outline from input text"""

        self.log.debug(f"outlining text(s)")

        if not texts:
            return ValueError("no files supplied")

        contents = "\n".join(
            map(lambda x: f"<file_contents>\n{x.strip()}\n</file_contents>", texts)
        )

        user_instructions = (
            instructions if instructions else "No instructions provided."
        )

        prompt = CONSPECT_SYSTEM_PROMPT.format(
            contents=contents, user_instructions=user_instructions
        )

        return await self.model.invoke_simple(
            [Message(Role.USER, prompt)], quality_mode=True
        )

    async def make_title(self, text: str) -> str:
        """Creates a title for input text"""

        self.log.debug(f"creating title for a text")

        messages = [
            Message(Role.SYSTEM, NAMING_PROMPT),
            Message(Role.USER, text.strip()),
        ]

        return await self.model.invoke_simple(messages)

    async def make_test(
        self, contents: list[str], explanation: str | None = None
    ) -> Test:
        self.log.debug(f"creating a test")

        contents = "\n".join(map(lambda x: f"<text>\n{x.strip()}\n</text>", contents))
        prompt = TEST_PROMPT.format(contents=contents, user_instructions=explanation)

        return await self.model.invoke_parsed(
            [Message(Role.USER, prompt)], Schema=Test, quality_mode=True
        )
