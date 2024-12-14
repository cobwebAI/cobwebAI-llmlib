from typing import Iterable
from uuid import UUID
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from enum import StrEnum
from dataclasses import dataclass


class ChatRole(StrEnum):
    USER = "user"
    BOT = "bot"


@dataclass
class Message:
    role: ChatRole
    raw_text: str
    attachment: str | None


@dataclass
class ChatAttachment:
    id: UUID
    content: str


type BotResponse = Message
type UserMessage = Message


class Chat:

    SYS_PROMPT = SystemMessage(
        (
            "You are an assistant for question-answering tasks. "
            "You will be provided with retrieved context to answer the questions. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answers concise. "
        )
    )

    CONTEXTUAL_PROMPT = HumanMessage(
        (
            "You are provided with additional context: "
            '"""\n{context}\n"""\n\n'
            "Use the context to respond to the user's prompt: "
            "{user_input}"
        )
    )

    def __init__(
        self, model: str = "gpt-4o-mini", oai_client: AsyncOpenAI | None = None
    ) -> None:

        self.log = logger
        self.chat = ChatOpenAI(model=model, async_client=oai_client)

        self.contextfull_prompt = ChatPromptTemplate.from_messages(
            [self.CONTEXTUAL_PROMPT]
        )
        self.regular_prompt = ChatPromptTemplate.from_messages(
            [HumanMessage("{user_input}")]
        )

    def _cast_to_human_msgs(self, message: Message) -> list[HumanMessage] | None:
        user_input = message.raw_text.strip()

        if not user_input or message.role != ChatRole.USER:
            return None

        if message.attachment:
            return self.contextfull_prompt.invoke(
                {"context": message.attachment, "user_input": user_input}
            ).to_messages()
        else:
            return self.regular_prompt.invoke(user_input).to_messages()

    def _cast_messages(
        self,
        messages: list[Message],
    ) -> list[HumanMessage | AIMessage]:
        output = []

        for message in messages:
            if message.role == ChatRole.BOT:
                output.append(AIMessage(message.raw_text))
            elif message.role == ChatRole.USER:
                output.extend(self._cast_to_human_msgs(message))
            else:
                raise ValueError("Unknow message role")

        return output

    def attachments_to_str(self, attachments: Iterable[ChatAttachment]) -> str:
        return "\n\n".join(map(lambda x: x.content, attachments))

    # def _filter_new_attachments(
    #     self, history: list[Message], attachments: list[ChatAttachment]
    # ) -> Iterable[ChatAttachment]:
    #     if not history:
    #         return attachments

    #     attachments_ids = set([a.id for a in attachments])

    #     for message in filter(lambda m: m.role == ChatRole.SYSTEM, history):
    #         for att_id in filter(
    #             lambda aid: str(aid) in message.content, attachments_ids
    #         ):
    #             attachments_ids.remove(att_id)

    #     return filter(lambda x: x.id in attachments_ids, attachments)

    async def invoke_chat(
        self,
        message: Message,
        history: list[Message] = [],
    ) -> Message | None:
        if user_message := self._cast_to_human_msgs(message):
            history = [self.SYS_PROMPT]
            history.extend(self._cast_messages(history))
            history.extend(user_message)

            response = await self.chat.ainvoke(history)

            return Message("bot", str(response.content), attachment=None)

        return None
