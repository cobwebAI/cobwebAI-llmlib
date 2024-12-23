from typing import Type
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI
from semchunk import chunkerify
from pydantic import BaseModel


class Question(BaseModel):
    question: str
    correct_answer: str
    incorrect_answers: list[str]
    correct_answer_explanation: str


class Test(BaseModel):
    test_name: str
    questions: list[Question]


class TextPostProcessing:
    """Corrects speech recognition mistakes in a text chunk by chunk"""

    CHUNK_SIZE = 3072
    """Text will be split in chunks each this number of tokens at max"""

    SYSTEM_PROMPT = (
        "Your job is to edit chunks of text produced by speech recognition system. "
        "The most important task here is to replace misrecognized terms, abbreviations and sequences of words with correct ones. "
        "You should also remove off-topic and time-wasting text, but keep important information as complete as possible. "
        "Respond only with edited text in the same language as the input chunk."
    )

    THEME_PROMPT = "You are provided with a probably correct approximate theme/description to guide you: {theme}."

    PREV_CHUNK_PROMPT = 'The previous chunk was already done for you: """\n{chunk}\n"""'

    CONSPECT_SYSTEM_PROMPT = """\
You are tasked with examining files and generating concise, well-organized summaries based on their content. Follow these 
instructions carefully:
1. Here are the contents of the files to be summarized:
{contents}

2. Your task is to create a summary of the information contained in these files. As you summarize:
   - Focus on the main ideas and key points
   - Organize the information in a logical and coherent manner
   - Be concise while ensuring all important details are included
   - If there are multiple files, identify common themes or connections between them
   - You should elaborate on terms and definitions, as well as add formulas and equations

3. The user has provided the following specific instructions for summarizing:
<user_instructions>
{user_instructions}
</user_instructions>
Please incorporate these instructions into your summarization process if they are not empty.

4. Format your output using markdown. For mathematical expressions:
   - Use $...$ for inline expressions (e.g., $x^2 + y^2$)
   - Use $$...$$ for block expressions

5. Ensure that your summary is well-structured, using appropriate markdown headers, lists, and other formatting as needed to enhance readability.

6. Respond only with your summary in the same language as the input text.
"""

    NAMING_PROMPT = (
        "You task is to create an appropriate descriptive title of the text provided by user. "
        "Respond only with a title in the same language as the input text."
    )

    TEST_PROMPT = """\
You are tasked with creating a multiple-choice test based on the content of a given text/texts. \
Your goal is to assess the user's knowledge and understanding of the text's contents. Here's what you'll be working with:
{contents}

The user has also provided the following to guide you in test creation:
<user_instructions>
{user_instructions}
</user_instructions>

Please follow these instructions to create the multiple-choice test:

1. Carefully read and analyze the provided text/texts.

2. Create a name for a test that accurately reflects the content of the text and the nature of the test.

3. Generate as many questions as possible based on the text. Each question should:
   a. Be clear and concise
   b. Test understanding of key concepts, facts, or ideas from the text
   c. Have one correct answer and 2-4 incorrect answers
   d. Include an explanation for why the correct answer is right and why the incorrect answers are wrong

4. Ensure that your questions cover a variety of topics from the text and use different types of questions (e.g., factual recall, inference, application of concepts).

5. Avoid creating questions that are too similar to each other or that give away answers to other questions.

6. Make sure that the incorrect answers are plausible but clearly incorrect when compared to the text.

7. Write clear and informative explanations for each correct answer, referencing specific parts of the text where applicable.

8. Format your output strictly according to the provided schema. Do not include any additional information or explanations outside of the schema structure.

Provided schema:
- test_name: An apporpriate name for the test;
- questions: List of objects of type Question.
The Question object has the following schema:
- question: A string representing a question itself;
- correct_answer: Correct answer to the question;
- incorrect_answers: List of incorrect answers;
- correct_answer_explanation: Explanation on why the correct_answer is correct and incorrect_answers are not.
"""

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

    async def _invoke_llm_simple(self, messages: list[dict[str, str]]) -> str:
        response = await self.oai_client.chat.completions.create(
            model=self.MODEL, messages=messages, n=1
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("text processing: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("text processing: model refusal")

        return response.choices[0].message.content

    async def _invoke_llm_parsed(
        self, messages: list[dict[str, str]], resp_format: Type[BaseModel]
    ) -> Type[BaseModel]:
        response = await self.oai_client.beta.chat.completions.parse(
            model=self.MODEL, messages=messages, n=1, response_format=resp_format
        )

        if response.choices[0].finish_reason != "stop":
            raise RuntimeError("test creation: unsuccessfull generation")
        elif response.choices[0].message.refusal:
            raise RuntimeError("test creation: model refusal")

        if output := response.choices[0].message.parsed:
            return output

        raise RuntimeError("test creation: failed to parse response")

    def _chunking_system_prompt(
        self,
        prev_chunk: str | None = None,
        theme: str | None = None,
    ) -> str:
        """Instantiates system prompt template with supplied info"""

        return "\n".join(
            [
                self.SYSTEM_PROMPT,
                self.THEME_PROMPT.format(theme=theme) if theme else "",
                self.PREV_CHUNK_PROMPT.format(chunk=prev_chunk) if prev_chunk else "",
            ]
        )

    async def fix_transcribed_text(self, text: str, theme: str | None = None) -> str:
        """Post-processes supplied text"""

        self.log.debug(f"post-processing text: {theme if theme else "no theme"}")

        if not text.strip():
            raise ValueError("no text supplied")

        if theme:
            theme = theme.strip()

        input_chunks = self.splitter.chunk(text)
        output_chunks: list[str] = []

        for i, current_chunk in enumerate(input_chunks):
            prev_chunk = output_chunks[i - 1] if i > 0 else None

            system_prompt = self._chunking_system_prompt(
                prev_chunk=prev_chunk,
                theme=(theme if theme else None),
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": current_chunk,
                },
            ]

            output_chunk = await self._invoke_llm_simple(messages)
            output_chunks.append(output_chunk)

        return " ".join(output_chunks)

    async def create_conspect_multi(
        self, texts: list[str], instructions: str | None = None
    ) -> str:
        """Creates an outline from input text"""

        self.log.debug(f"outlining text(s)")

        if not texts:
            return ValueError("no files supplied")

        contents = "\n".join(
            map(lambda x: f"<file_contents>\n{x.strip()}\n</file_contents>", texts)
        )

        user_instructions = (
            instructions if instructions else "No instructions provided."
        )

        prompt = self.CONSPECT_SYSTEM_PROMPT.format(
            contents=contents, user_instructions=user_instructions
        )

        return await self._invoke_llm_simple([{"role": "user", "content": prompt}])

    async def make_title(self, text: str) -> str:
        """Creates a title for input text"""

        self.log.debug(f"creating title for a text")

        messages = [
            {
                "role": "system",
                "content": self.NAMING_PROMPT,
            },
            {
                "role": "user",
                "content": text.strip(),
            },
        ]

        return await self._invoke_llm_simple(messages)

    async def make_test(
        self, contents: list[str], explanation: str | None = None
    ) -> Test:
        self.log.debug(f"creating a test")

        contents = "\n".join(map(lambda x: f"<text>\n{x.strip()}\n</text>", contents))

        prompt = self.TEST_PROMPT.format(
            contents=contents, user_instructions=explanation
        )

        return await self._invoke_llm_parsed(
            [{"role": "user", "content": prompt}], resp_format=Test
        )
