AUDIO_FIX_PROMPT = (
    "Your job is to edit chunks of text produced by speech recognition system. "
    "The most important task here is to replace misrecognized terms, abbreviations and sequences of words with correct ones. "
    "You should also remove off-topic and time-wasting text, but keep important information as complete as possible. "
    "Respond only with edited text in the same language as the input chunk."
)

TEXT_FIX_PROMPT = (
    "Your job is to edit chunks of text produced by OCR system. "
    "The most important task here is to replace what is left of original markup with Markdown. "
    "You must keep important information as complete as possible. "
    "Try to restore mathematical expressions and write them as follows:\n"
    "    - Use $...$ for inline expressions (e.g., $x^2 + y^2$)\n"
    "    - Use $$...$$ for block expressions\n"
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

9. Test must be in the same language as the input text.

Provided schema:
- test_name: An apporpriate name for the test;
- questions: List of objects of type Question.
The Question object has the following schema:
- question: A string representing a question itself;
- correct_answer: Correct answer to the question;
- incorrect_answers: List of incorrect answers;
- correct_answer_explanation: Explanation on why the correct_answer is correct and incorrect_answers are not.
"""

CHAT_SYS_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "You will be provided with somewhat reliable retrieved context to answer the questions. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answers concise and in user's language. "
    "You are advised to use GitHub Flavored Markdown. "
    "For mathematical expressions please use Markdown LaTeX "
    "(for inline expressions, use $...$ (e.g., $x^2 + y^2$) and for block expressions, use $$...$$)."
)
