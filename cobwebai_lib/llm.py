from uuid import UUID
from openai import AsyncOpenAI
from .vdb import VectorDB
from .audio import Transcription
from .text import TextPostProcessing
from .chat import Chat, ChatAttachment, Message, UserMessage, BotResponse


class LLMTools:
    OAI_MODEL = "gpt-4o-mini"
    OAI_EMBED_MODEL = "text-embedding-ada-002"

    ATTACHMENT_MAX_LEN = 1024

    def __init__(self, api_key: str | None = None, chroma_port: int = 35432) -> None:
        self.common_oai_client = AsyncOpenAI(api_key=api_key)
        self.s2t = Transcription(self.common_oai_client)
        self.s2t_pp = TextPostProcessing(self.common_oai_client)
        self.chat = Chat(self.OAI_MODEL, oai_key=api_key)
        self.vdb = VectorDB(
            embed_model_name=self.OAI_EMBED_MODEL,
            chroma_port=chroma_port,
            oai_key=api_key,
        )

    async def _process_attachments(
        self,
        user_id: UUID,
        project_id: UUID,
        attachments: list[ChatAttachment],
        user_prompt: str,
    ) -> str | None:
        short_atts: list[ChatAttachment] = []

        for att in attachments:
            if len(att.content) <= self.ATTACHMENT_MAX_LEN:
                short_atts.append(att.content)
                continue

            if vdb_out := await self.vdb.store_and_retrieve(
                user_id, project_id, att.id, att.content, user_prompt
            ):
                short_atts.extend(vdb_out)

        return "\n\n".join(short_atts) if short_atts else None

    async def chat_with_rag(
        self,
        user_id: UUID,
        project_id: UUID,
        user_prompt: str,
        attachments: list[ChatAttachment] = [],
        history: list[Message] = [],
    ) -> tuple[UserMessage, BotResponse] | tuple[UserMessage, None] | None:
        user_prompt = user_prompt.strip()

        if not user_prompt:
            return None

        user_msg = Message(
            role="user",
            raw_text=user_prompt,
            attachment=await self._process_attachments(
                user_id, project_id, attachments, user_prompt
            ),
        )

        return (user_msg, await self.chat.invoke_chat(user_msg, history))
