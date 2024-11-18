from logging import Logger
from openai import AsyncOpenAI
from cobwebAI_llmlib import logger


class TextProcessing:
    def __init__(
        self,
        oai_client: AsyncOpenAI | None = None,
        log: Logger | None = None,
        **kwargs,
    ) -> None:
        self.log = log if log else logger.log
        self.oai_client = oai_client if oai_client else AsyncOpenAI(**kwargs)

    async def fix_transcribed_text(self, text: str, theme: str | None = None) -> str:
        system_prompt = "You are tasked with post-processing speech to text output. \
            Your most important task is to replace wrongly recognized terms, abbreviations and sequences of words with correct ones. \
            Your secondary task is to remove possible off-topic text and artifacts of speech to text process. \
            You must not shorten on-topic content."

        extra_prompt = f"Additionaly, you are provided with a theme of the text to guide you: {theme}."

        format_prompt = "Respond only with post-processed text in the same language as the input text."

        response = await self.oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"{system_prompt} {extra_prompt} {format_prompt}",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content
