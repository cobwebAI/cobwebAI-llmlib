from loguru import logger
from openai import AsyncOpenAI
from semchunk import chunkerify


class TextPostProcessing:
    """Corrects speech recognition mistakes in a text chunk by chunk"""

    CHUNK_SIZE = 3072
    """Text will be split in chunks each this number of tokens at max"""

    SYSTEM_PROMPT = (
        "Your job is to edit chunks of text produced by speech recognition system. "
        "The most important task here is to replace misrecognized terms, abbreviations and sequences of words with correct ones. "
        "You should also remove off-topic and time-wasting text, but keep important information as complete as possible. "
        "Respond only with edited text in the same language as the input chunk."
    )

    THEME_PROMPT = "You are provided with a probably correct approximate theme/description to guide you: {theme}."

    PREV_CHUNK_PROMPT = 'The previous chunk was already done for you: """\n{chunk}\n"""'

    CONSPECT_SYSTEM_PROMPT = (
        "Your task is to extract as much knowledge as possible from user's text by creating something like lecture notes. "
        "If input text is too short for that task, just rewrite it to be more readable. "
        "You are advised to use GitHub Flavored Markdown. "
        "You should elaborate on terms and definitions, as well as add formulas and equations, "
        "but only if you are confident in your correctness."
        "Please use the Markdown LaTeX syntax for mathematical expressions "
        "(for inline expressions, use $...$ (e.g., $x^2 + y^2$) and for block expressions, use $$...$$). "
        "Respond only with output text in the same language as the input text."
    )

    MULTITEXT_PROMPT = (
        'Mutliple texts are supplied, separated for your convenience by "==TEXT SEPARATOR==", '
        "take that into consideration when writing headings."
    )

    NAMING_PROMPT = (
        "You task is to create an appropriate descriptive title of the text provided by user. "
        "Respond only with a title in the same language as the input text."
    )

    MODEL = "gpt-4o-mini"

    def __init__(
        self,
        oai_client: AsyncOpenAI | None = None,
        **kwargs,
    ) -> None:
        """Constructs text post-processor with provided resources or its own"""

        self.log = logger
        self.oai_client = oai_client if oai_client else AsyncOpenAI(**kwargs)

        # Shoud I disable memoization here?
        self.splitter = chunkerify(
            self.MODEL, chunk_size=self.CHUNK_SIZE, memoize=False
        )

    async def _invoke_llm_simple(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        response = await self.oai_client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            n=1,
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content

    def _chunking_system_prompt(
        self,
        prev_chunk: str | None = None,
        theme: str | None = None,
    ) -> str:
        """Instantiates system prompt template with supplied info"""

        return "\n".join(
            [
                self.SYSTEM_PROMPT,
                self.THEME_PROMPT.format(theme=theme) if theme else "",
                self.PREV_CHUNK_PROMPT.format(chunk=prev_chunk) if prev_chunk else "",
            ]
        )

    async def fix_transcribed_text(self, text: str, theme: str | None = None) -> str:
        """Post-processes supplied text"""

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")

        if not text.strip():
            raise ValueError("no text supplied")

        if theme:
            theme = theme.strip()

        input_chunks = self.splitter.chunk(text)
        output_chunks: list[str] = []

        for i, current_chunk in enumerate(input_chunks):
            prev_chunk = output_chunks[i - 1] if i > 0 else None

            system_prompt = self._chunking_system_prompt(
                prev_chunk=prev_chunk,
                theme=(theme if theme else None),
            )

            output_chunk = await self._invoke_llm_simple(system_prompt, current_chunk)
            output_chunks.append(output_chunk)

        return " ".join(output_chunks)

    async def create_conspect_multi(
        self, texts: list[str], theme: str | None = None
    ) -> str:
        """Creates an outline from input text"""

        self.log.debug(f"outlining multiple texts: {theme if theme else "no theme"}")

        if not texts:
            return ValueError("no files supplied")

        system_prompt = (
            f"{self.CONSPECT_SYSTEM_PROMPT}/n{self.THEME_PROMPT.format(theme=theme)}/n"
            if theme
            else self.CONSPECT_SYSTEM_PROMPT
        )

        if len(texts) > 1:
            system_prompt += self.MULTITEXT_PROMPT

        text = "\n\n==TEXT SEPARATOR==\n\n".join(map(lambda x: x.strip(), texts))

        return await self._invoke_llm_simple(system_prompt, text)

    async def make_title(self, text: str) -> str:
        """Creates a title for input text"""

        self.log.debug(f"creating title for a text")

        return await self._invoke_llm_simple(self.NAMING_PROMPT, text.strip())
