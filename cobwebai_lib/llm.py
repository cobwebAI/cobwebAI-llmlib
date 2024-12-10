from openai import AsyncOpenAI
from .vdb import VectorDB
from .audio import Transcription
from .text import TextPostProcessing


class LLMTools:
    OAI_MODEL = "gpt-4o-mini"
    OAI_EMBED_MODEL = "text-embedding-ada-002"

    def __init__(self, api_key: str | None = None, chroma_port: int = 35432) -> None:
        self.oai_client = AsyncOpenAI(api_key=api_key)
        self.vdb = VectorDB(
            embed_model_name=self.OAI_EMBED_MODEL,
            chroma_port=chroma_port,
            oai_key=api_key,
        )
        self.s2t = Transcription(self.oai_client)
        self.s2t_pp = TextPostProcessing(self.oai_client)
