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

    THEME_PROMPT = "You are provided with a probably correct approximate theme/description to guide you: {theme}."

    PREV_CHUNK_PROMPT = 'The previous chunk is already done for you: """\n{chunk}\n"""'

    FINAL_PROMPT = (
        "That is the end of your instructions. Get ready to recieve input text."
    )

    CONSPECT_SYSTEM_PROMPT = (
        "Your task is to extract as much knowledge as possible from user's text by creating something like lecture notes. "
        "If input text is too short for that task, just rewrite it to be more readable. "
        "You are advised to use markdown, with LaTeX there (using $ symbols). "
        "If you are confident, you should elaborate on terms and definitions, as well as add formulas and equations too. "
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
        response = await self.oai_client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content

    def _chunking_system_prompt(
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
    ) -> str:
        """Post-processes supplied text"""

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")
        
        if not text.strip():
            raise ValueError("no text supplied")

        input_chunks: list[str] = []
        output_chunks: list[str] = []

        if chunk_size is None:
            input_chunks = self.splitter.chunk(text)
        else:
            input_chunks = chunkerify(
                self.MODEL, chunk_size=chunk_size, memoize=False
            ).chunk(text)

        for input_chunk in input_chunks:
            system_prompt = self._chunking_system_prompt(
                previous_chunk=(output_chunks[-1] if output_chunks else None),
                theme=(theme if theme else None),
            )

            output_chunks.append(
                await self._invoke_llm_simple(system_prompt, input_chunk)
            )

        return " ".join(output_chunks)
    
    async def create_conspect_multi(self, texts: list[str], theme: str | None = None) -> str:
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
