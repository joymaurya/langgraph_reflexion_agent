from dotenv import load_dotenv
load_dotenv()
import json
from typing import List,DefaultDict
from schemas import AnswerQuestion,Reflection
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from chains import parser
from langchain_core.messages import BaseMessage,ToolMessage,HumanMessage,AIMessage

tavily_api_wrapper=TavilySearchAPIWrapper()
tavily_tool=TavilySearchResults(api_wrapper=tavily_api_wrapper,max_results=5)


def execute_tools(State:List[BaseMessage])->List[ToolMessage]:
    tool_invocation=State[-1]
    parsed=parser.invoke(tool_invocation)
    output=[]
    search_queries=[]
    ids=[]
    for parse_call in parsed:
        for obj in parse_call["args"]["search_queries"]:   
            output.append(tavily_tool.run(obj))
            search_queries.append(obj)
            ids.append(parse_call["id"])

    answer=DefaultDict(dict)

    for id,search,out in zip(ids,search_queries,output):
        per_query_content=[]
        for obj in out:
            per_query_content.append({"content":obj.get("content")})
        answer[id][search]=per_query_content

    tool_messages=[]
    for key,value in answer.items():
        tool_messages.append(ToolMessage(content=json.dumps(value),tool_call_id=key))
    return tool_messages

if __name__=="__main__":

    human_message=HumanMessage("Write a artical about autonomous SOC")

    answer=AnswerQuestion(
        answer="",
        reflection=Reflection(missing="",superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups"
        ],
    )

    execute_tools(
        [
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name":AnswerQuestion.__name__,
                        "args":answer.model_dump(),
                        "id":"XYZ"
                    }
                ]
            )
        ]
    )