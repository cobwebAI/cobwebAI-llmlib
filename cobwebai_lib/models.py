from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Type
from enum import StrEnum

from pydantic import BaseModel

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic, NOT_GIVEN


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    BOT = "bot"


@dataclass
class Message:
    role: Role
    content: str
    attachment: str | None = None


class LanguageModel(ABC):
    @abstractmethod
    def invoke_simple(self, messages: list[Message], quality_mode: bool = False) -> str:
        pass

    @abstractmethod
    def invoke_parsed(
        self,
        messages: list[Message],
        Schema: Type[BaseModel],
        quality_mode: bool = False,
    ) -> BaseModel:
        pass

    def _cast_msg(self, msg: Message) -> dict[str, str]:
        role = None

        match msg.role:
            case Role.SYSTEM:
                role = "system"
            case Role.USER:
                role = "user"
            case Role.BOT:
                role = "assistant"

        return {"role": role, "content": msg.content}


class OAIModel(LanguageModel):
    MODEL = "gpt-4o-mini"
    QUALITY_MODEL = "gpt-4o"

    def __init__(self, oai_client: AsyncOpenAI | None = None, **kwargs) -> None:
        super().__init__()
        self.client = oai_client if oai_client else AsyncOpenAI(**kwargs)

    async def invoke_simple(
        self, messages: list[Message], quality_mode: bool = False
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.MODEL if not quality_mode else self.QUALITY_MODEL,
            messages=list(map(self._cast_msg, messages)),
            n=1,
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content

    async def invoke_parsed(
        self,
        messages: list[Message],
        Schema: Type[BaseModel],
        quality_mode: bool = False,
    ) -> BaseModel:
        response = await self.client.beta.chat.completions.parse(
            model=self.MODEL if not quality_mode else self.QUALITY_MODEL,
            messages=list(map(self._cast_msg, messages)),
            n=1,
            response_format=Schema,
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("test creation: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("test creation: model refusal")

        if output := response.choices[0].message.parsed:
            return output
        else:
            raise RuntimeError("test creation: failed to parse response")


class AnthModel(LanguageModel):
    MODEL = "claude-3-7-sonnet-latest"
    QUALITY_MODEL = MODEL
    MAX_TOKENS = 32 * 1024

    def __init__(self, ant_client: AsyncAnthropic | None = None, **kwargs) -> None:
        super().__init__()

        self.tool_use_reminder = "\n\nYou must send the results via send_result tool."
        self.client = ant_client if ant_client else AsyncAnthropic(**kwargs)

    def _strip_system(self, messages: list[Message]) -> tuple[str, list[Message]]:
        if messages[0].role == Role.SYSTEM:
            return (messages[0].content, messages[1:])
        else:
            return (None, messages)

    async def invoke_simple(
        self, messages: list[Message], quality_mode: bool = False
    ) -> str:
        sys_prompt, messages = self._strip_system(messages)

        response = None

        async with self.client.messages.stream(
            max_tokens=self.MAX_TOKENS,
            messages=list(map(self._cast_msg, messages)),
            model=self.MODEL if not quality_mode else self.QUALITY_MODEL,
            system=sys_prompt if sys_prompt else NOT_GIVEN,
        ) as stream:
            response = await stream.get_final_message()

        return response.content[0].text

    async def invoke_parsed(
        self,
        messages: list[Message],
        Schema: Type[BaseModel],
        quality_mode: bool = False,
    ) -> BaseModel:
        sys_prompt, messages = self._strip_system(messages)
        sys_prompt += self.tool_use_reminder

        response = None

        async with self.client.messages.stream(
            max_tokens=self.MAX_TOKENS,
            messages=list(map(self._cast_msg, messages)),
            model=self.MODEL if not quality_mode else self.QUALITY_MODEL,
            system=sys_prompt if sys_prompt else NOT_GIVEN,
            tool_choice={"type": "tool", "name": "send_result"},
            tools=[
                {
                    "name": "send_result",
                    "description": "Send the result",
                    "input_schema": Schema.model_json_schema(),
                }
            ],
        ) as stream:
            response = await stream.get_final_message()

        return Schema(**response.content[1].input)
