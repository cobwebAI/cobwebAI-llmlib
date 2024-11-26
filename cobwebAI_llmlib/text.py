from logging import Logger
from openai import AsyncOpenAI
from cobwebai_llmlib import logger
from semchunk import chunkerify


class TextPostProcessing:
    """Corrects speech recognition mistakes in a text chunk by chunk"""

    CHUNK_SIZE = 3072
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

    def __init__(
        self,
        oai_client: AsyncOpenAI | None = None,
        log: Logger | None = None,
        **kwargs,
    ) -> None:
        """Constructs text post-processor with provided resources or its own"""

        self.log = log if log else logger.log
        self.oai_client = oai_client if oai_client else AsyncOpenAI(**kwargs)

        # Shoud I disable default cache here?
        self.splitter = chunkerify("gpt-4o-mini", chunk_size=self.CHUNK_SIZE)

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

        return f"{self.SYSTEM_PROMPT}\n{theme_prompt}\n{prev_prompt}\n"

    async def fix_transcribed_text(
        self, text: str, theme: str | None = None
    ) -> str | None:
        """Post-processes supplied text"""

        output_chunks: list[str] = []

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")

        try:
            for input_chunk in self.splitter.chunk(text):
                system_prompt = self.build_system_prompt(
                    previous_chunk=(
                        output_chunks[-1] if len(output_chunks) > 0 else None
                    ),
                    theme=(theme if theme else None),
                )

                response = await self.oai_client.chat.completions.create(
                    model="gpt-4o-mini",
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

        except Exception as e:
            self.log.error(f"fix_transcribed_text failed with: {e}")

        return " ".join(output_chunks)
