 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv,find_dotenv
import os
load_dotenv(find_dotenv())

# model = OpenAI(model_name="gpt-3.5-turbo-instruct", )

model = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.0, 
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @model_validator(mode="before")
    @classmethod
    def aaa(cls, values: dict) -> dict:
        print("Validating...")
        # print(values)
        setup = values.get("setup")
        if setup and setup[-1] != ".":
            raise ValueError("Badly formed question!")
        return values
    # def question_ends_with_question_mark(cls, values: dict) -> dict:
    #     print("Validating...")
    #     # print(values)
    #     setup = values.get("setup")
    #     if setup and setup[-1] != ".":
    #         raise ValueError("Badly formed question!")
    #     return values


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "Tell me a joke."})

import pdb; pdb.set_trace()
result = parser.invoke(output)
# print(result)