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
        self.oai_client = AsyncOpenAI(api_key=api_key)
        self.vdb = VectorDB(
            embed_model_name=self.OAI_EMBED_MODEL,
            chroma_port=chroma_port,
            oai_key=api_key,
        )
        self.s2t = Transcription(self.oai_client)
        self.s2t_pp = TextPostProcessing(self.oai_client)
        self.chat = Chat(self.OAI_MODEL, self.oai_client)

    async def chat_with_rag(
        self,
        user_id: UUID,
        project_id: UUID,
        user_message: str,
        attachments: list[ChatAttachment] = [],
        history: list[Message] = [],
    ) -> tuple[UserMessage, BotResponse] | tuple[UserMessage, None] | None:

        user_message = user_message.strip()

        if not user_message:
            return None

        norm_attaches: list[ChatAttachment] = []

        for attachment in attachments:
            if len(attachment.content <= self.ATTACHMENT_MAX_LEN):
                norm_attaches.append(attachment)
                continue

            vdb_out = self.vdb.store_and_retrieve(
                user_id, project_id, attachment.id, attachment.content, user_message
            )

            if vdb_out:
                norm_attaches.append(ChatAttachment(attachment.id, vdb_out[0]))

        attached = None

        if norm_attaches:
            attached = self.chat.attachments_to_str(norm_attaches)

        user_msg = Message(
            role="user",
            raw_text=user_message,
            attachment=attached,
        )

        response = await self.chat.invoke_chat(user_msg, history)

        return (user_message, response)
