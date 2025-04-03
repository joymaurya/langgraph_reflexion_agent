from langchain_openai import ChatOpenAI
import datetime
from schemas import AnswerQuestion,RevisedAnswer
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

parser=JsonOutputToolsParser(return_id=True)
parser_pydantic=PydanticToolsParser(tools=[AnswerQuestion])
llm=ChatOpenAI(model="gpt-4-turbo-preview")

actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are expert researcher.
            Current time:{time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question using the required format.")
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

revise_instructions="""
    Revise your previous answer with new information.
    - Gather new information using previous critique and queries and add relevent information.
    - Remove superfluous data which was mentioned.
    - You must include numerical citations in your answer to ensure it can be verified
    - Add a "Reference" Section to the bottom of the answer (Which does not count in word limit) In form of
       1) https:examples.com
       2) https:examples.com
"""

first_responder_prompt_template=actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder=first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion],tool_choice="AnswerQuestion"
)

revised_response=actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(
    tools=[RevisedAnswer],tool_choice="RevisedAnswer"
)

if __name__=="__main__":
    human_message=HumanMessage(
        content="Write about AI-Powered SDC / automated soc problem domain."
    )
    chain=(
        first_responder|parser_pydantic
    )
    res=chain.invoke({"messages":[human_message]})