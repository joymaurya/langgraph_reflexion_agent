from langgraph.graph import END,MessageGraph
from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage,ToolMessage,HumanMessage
load_dotenv()
from chains import first_responder,revised_response
from tool_execution import execute_tools

builder=MessageGraph()

builder.add_node("draft",first_responder)

builder.add_node("execute_tools",execute_tools)

builder.add_node("revise",revised_response)

builder.add_edge("draft","execute_tools")

builder.add_edge("execute_tools","revise")

def decision(state:List[BaseMessage]):
    count_tools_visits=sum(isinstance(message,ToolMessage) for message in state)
    if count_tools_visits>2:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise",decision)
builder.set_entry_point("draft")

graph=builder.compile()

if __name__=="__main__":
    human_message=HumanMessage(
        content="Write about AI-Powered SDC / automated soc problem domain."
    )
    resp=graph.invoke("Write about AI-Powered SDC / automated soc problem domain.")
    print(resp)