from os import environ
from uuid import UUID
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dataclasses import dataclass
from .models import Role, Message
from .prompts import CHAT_SYS_PROMPT


@dataclass
class ChatAttachment:
    id: UUID
    content: str


class Chat:
    CONTEXTUAL_PROMPT = (
        "You are provided with a part of context: "
        '"""\n{context}\n"""\n\n'
        "You can use all parts of context to respond to the user's prompt: "
        "{user_input}"
    )

    def __init__(self, model: str = "claude-3-7-sonnet-latest") -> None:

        self.log = logger
        self.sys_msg = SystemMessage(CHAT_SYS_PROMPT)

        if model.startswith("gpt"):
            self.chat = ChatOpenAI(model=model)
        else:
            self.chat = ChatAnthropic(model=model)

    def _cast_user_msg(self, message: Message) -> list[HumanMessage]:
        user_input = message.raw_text.strip()
        context = message.attachment.strip() if message.attachment else None

        if not user_input or message.role != Role.USER:
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
            if message.role == Role.BOT:
                output.append(AIMessage(message.raw_text))
            elif message.role == Role.USER:
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
            history_actual = [self.sys_msg]
            history_actual.extend(self._cast_messages(history))
            history_actual.extend(user_message)

            try:
                response = await self.chat.ainvoke(history_actual)
                return Message(Role.BOT, str(response.content))
            except Exception as e:
                self.log.error(f"Failed to invoke chat: {e}")

        return None
