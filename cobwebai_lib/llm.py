from uuid import UUID
from loguru import logger
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import tiktoken
from .vdb import VectorDB
from .audio import Transcription
from .text import Test, TextPostProcessing
from .chat import Chat, ChatAttachment, Message

type BotResponse = Message
type UserMessage = Message


class LLMTools:
    CHAT_MODEL = "claude-3-7-sonnet-latest"

    OAI_EMBED_MODEL = "text-embedding-ada-002"
    OAI_TOKENIZER = "o200k_base"
    OAI_CONTEXT_TOKEN_LIMIT = 32768

    ATTACHMENT_RAG_CUTOFF_LEN = 4096
    FORCED_ATTACHMENT_MAX_LEN = 8192
    MAX_LEN_FOR_TEST = 30000

    def __init__(
        self,
        oai_api_key: str | None = None,
        chroma_port: int = 35432,
        chroma_host: str = "localhost",
        force_openai: bool = False,
    ) -> None:
        self.log = logger
        self.tokenizer = tiktoken.get_encoding(self.OAI_TOKENIZER)
        self.oai_client = AsyncOpenAI()
        self.anth_client = AsyncAnthropic()

        self.s2t = Transcription(self.oai_client if force_openai else None)

        self.s2t_pp = TextPostProcessing(
            self.oai_client if force_openai else self.anth_client
        )

        self.chat = Chat(self.CHAT_MODEL)

        self.vdb = VectorDB(
            embed_model_name=self.OAI_EMBED_MODEL,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            oai_key=oai_api_key,
        )

    async def _process_attachments(
        self,
        user_id: UUID,
        project_id: UUID,
        attachments: list[ChatAttachment],
        rag_attachments: list[ChatAttachment],
        user_prompt: str | None = None,
    ) -> str | None:
        short_atts: list[ChatAttachment] = []

        if user_prompt:
            user_prompt = user_prompt.strip()

        async def retriever(doc_id: UUID, content: str) -> list[str]:
            if not user_prompt:
                return []

            if vdb_out := await self.vdb.store_and_retrieve(
                user_id, project_id, doc_id, content, user_prompt
            ):
                return vdb_out
            else:
                self.log.info("couldn't retrieve context from vdb")
                return []

        for att in attachments:
            if len(att.content) <= self.FORCED_ATTACHMENT_MAX_LEN:
                short_atts.append(att.content)
            else:
                short_atts.extend(await retriever(att.id, att.content))

        for ratt in rag_attachments:
            if len(ratt.content) <= self.ATTACHMENT_RAG_CUTOFF_LEN:
                short_atts.append(ratt.content)
            else:
                short_atts.extend(await retriever(ratt.id, ratt.content))

        return "\n\n".join(short_atts) if short_atts else None

    async def chat_with_rag(
        self,
        user_id: UUID,
        project_id: UUID,
        user_prompt: str,
        attachments: list[ChatAttachment] = [],
        rag_attachments: list[ChatAttachment] = [],
        history: list[Message] = [],
    ) -> tuple[UserMessage, BotResponse] | tuple[UserMessage, None] | None:
        """Responds to user's prompt and attached context using automatic optional RAG

        Args:
            attachments (list[ChatAttachment], optional): Attachments to preferably pass as is. Defaults to [].
            rag_attachments (list[ChatAttachment], optional): Attachments to preferably pass through vector db. Defaults to [].
        """

        user_prompt = user_prompt.strip()

        if not user_prompt:
            return None

        user_msg = Message(
            role="user",
            content=user_prompt,
            attachment=await self._process_attachments(
                user_id, project_id, attachments, rag_attachments, user_prompt
            ),
        )

        self.log.info(
            f"chatting: user {user_id}, final attachment length {len(user_msg.attachment if user_msg.attachment else "")}"
        )

        return (user_msg, await self.chat.invoke_chat(user_msg, history))

    async def generate_note(
        self, files: list[str], custom_description: str | None = None
    ) -> tuple[str, str]:
        """Generates title and content of a note from specified files (texts).

        Args:
            cutom_description (str | None): theme/description of files.

        Returns:
            tuple[str, str]: theme and content of note.
        """

        content = await self.s2t_pp.create_conspect_multi(files, custom_description)
        title = await self.s2t_pp.make_title(content)

        return (title, content)

    async def transcribe_avfile(
        self, path: str, language: str = "ru", theme: str | None = None
    ) -> str:
        """Transcribes audio/video file with post-processing"""

        raw_text = await self.s2t.transcribe_file(path, language)
        fixed_text = await self.s2t_pp.fix_transcribed_text(raw_text, theme)
        return fixed_text

    async def _process_attachments_for_test(
        self, attachments: list[ChatAttachment]
    ) -> list[str]:
        output = []
        output_length = 0

        for attachment in attachments:
            tokens = self.tokenizer.encode(attachment.content)
            output_length += len(tokens)
            output.append(attachment.content)

            if output_length > self.OAI_CONTEXT_TOKEN_LIMIT:
                raise ValueError(
                    "attachments together exceeded specified context length"
                )
        return output

    async def create_test(
        self,
        user_id: UUID,
        project_id: UUID,
        explanation: str | None = None,
        attachments: list[ChatAttachment] = [],
    ) -> Test:
        context = await self._process_attachments_for_test(attachments)
        return await self.s2t_pp.make_test(context, explanation)
