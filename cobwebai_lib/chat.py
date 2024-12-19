from os import environ
from typing import Iterable
from uuid import UUID
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from enum import StrEnum
from dataclasses import dataclass
from typing_extensions import Annotated, TypedDict


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

# class Question(TypedDict):
#     """Test question and answer variants."""

#     question: Annotated[str, ..., "Test question"]
#     varinant_a: Annotated[str, ..., "Answer variant A"]
#     varinant_b: Annotated[str, ..., "Answer variant B"]
#     varinant_c: Annotated[str, ..., "Answer variant C"]
#     varinant_d: Annotated[str, ..., "Answer variant D"]


class Chat:

    SYS_MSG = SystemMessage(
        (
            "You are an assistant for question-answering tasks. "
            "You will be provided with somewhat reliable retrieved context to answer the questions. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answers concise and in user's language. "
            "You are advised to use GitHub Flavored Markdown. "
            "For mathematical expressions please use Markdown LaTeX "
            "(for inline expressions, use $...$ (e.g., $x^2 + y^2$) and for block expressions, use $$...$$)."
        )
    )

    CONTEXTUAL_PROMPT = (
        "You are provided with a part of context: "
        '"""\n{context}\n"""\n\n'
        "You can use all parts of context to respond to the user's prompt: "
        "{user_input}"
    )

    def __init__(self, model: str = "gpt-4o-mini", oai_key: str | None = None) -> None:

        self.log = logger
        self.chat = ChatOpenAI(
            model=model, api_key=(oai_key if oai_key else environ["OPENAI_API_KEY"])
        )

    def _cast_user_msg(self, message: Message) -> list[HumanMessage]:
        user_input = message.raw_text.strip()
        context = message.attachment.strip() if message.attachment else None

        if not user_input or message.role != ChatRole.USER:
            self.log.warning(f"{message} does not contain text or has a wrong role")
            return []

        return (
            [HumanMessage(user_input)]
            if not context
            else [
                HumanMessage(
                    self.CONTEXTUAL_PROMPT.format(
                        context=context,
                        user_input=user_input,
                    )
                )
            ]
        )

    def _cast_messages(
        self,
        messages: list[Message],
    ) -> list[HumanMessage | AIMessage]:
        output = []

        for message in messages:
            if message.role == ChatRole.BOT:
                output.append(AIMessage(message.raw_text))
            elif message.role == ChatRole.USER:
                output.extend(self._cast_user_msg(message))
            else:
                raise ValueError("Unknow message role")

        return output

    async def invoke_chat(
        self,
        message: Message,
        history: list[Message] = [],
    ) -> Message | None:
        if user_message := self._cast_user_msg(message):
            history_actual = [self.SYS_MSG]
            history_actual.extend(self._cast_messages(history))
            history_actual.extend(user_message)

            try:
                response = await self.chat.ainvoke(history_actual)
                return Message("bot", str(response.content), attachment=None)
            except Exception as e:
                self.log.error(f"Failed to invoke chat: {e}")

        return None

    # async def invoke_test_generation(self) -> list[Question]:
    #     self.chat.with_structured_output(Question, strict=True)
