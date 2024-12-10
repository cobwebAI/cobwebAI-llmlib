from loguru import logger
from openai import AsyncOpenAI
from semchunk import chunkerify


class TextPostProcessing:
    """Corrects speech recognition mistakes in a text chunk by chunk"""

    CHUNK_SIZE = 3500
    """Text will be split in chunks each this number of tokens at max"""

    SYSTEM_PROMPT = (
        "Your job is to edit chunks of text that come out of speech recognition system. "
        "The most important task here is to replace misrecognized terms, abbreviations and sequences of words with correct ones. "
        "You also need to remove text that is too off-topic and possible artifacts of speech recognition process. "
        "You should not shorten the text more than truly necessary. "
        "Respond only with edited text in the same language as the input chunk."
    )

    THEME_PROMPT = (
        "You are provided with a theme of the entire text to guide you: {theme}."
    )

    PREV_CHUNK_PROMPT = 'The previous chunk is already done for you: """\n{chunk}\n"""'

    FINAL_PROMPT = (
        "That is the end of your instructions. Get ready to recieve input text."
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

    def build_system_prompt(
        self,
        previous_chunk: str | None = None,
        theme: str | None = None,
    ) -> str:
        """Instantiates system prompt template with supplied info"""

        prev_prompt = self.PREV_CHUNK_PROMPT.format(chunk=previous_chunk)
        theme_prompt = self.THEME_PROMPT.format(theme=theme)

        if not previous_chunk:
            prev_prompt = ""

        if not theme:
            theme_prompt = ""

        return (
            f"{self.SYSTEM_PROMPT}\n{theme_prompt}\n{prev_prompt}\n{self.FINAL_PROMPT}"
        )

    async def fix_transcribed_text(
        self, text: str, theme: str | None = None, chunk_size: int | None = None
    ) -> str | None:
        """Post-processes supplied text"""

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")

        try:
            input_chunks: list[str] = []
            output_chunks: list[str] = []

            if chunk_size is None:
                input_chunks = self.splitter.chunk(text)
            else:
                input_chunks = chunkerify(
                    self.MODEL, chunk_size=chunk_size, memoize=False
                ).chunk(text)

            for input_chunk in input_chunks:
                system_prompt = self.build_system_prompt(
                    previous_chunk=(output_chunks[-1] if output_chunks else None),
                    theme=(theme if theme else None),
                )

                response = await self.oai_client.chat.completions.create(
                    model=self.MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": input_chunk,
                        },
                    ],
                )

                if response.choices[0].finish_reason != "stop":
                    raise RuntimeError("text processing: unsuccessfull generation")
                elif response.choices[0].message.refusal:
                    raise RuntimeError("text processing: model refusal")

                output_chunks.append(response.choices[0].message.content)

            return " ".join(output_chunks)

        except Exception as e:
            self.log.error(f"fix_transcribed_text failed with: {e}")

        return None
