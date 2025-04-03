from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str=Field(description="Critique of what is missing")
    superfluous:str=Field(description="Critique of what is superfluous")

class AnswerQuestion(BaseModel):
    answer:str=Field(description="~250 word detailed answer to the question")
    search_queries:List[str]=Field(
        description="1-3 search queries for researching improvments to address the critique of your current answer."
    )
    reflection:Reflection=Field(description="Your reflection on initial answer.")
    
class RevisedAnswer(AnswerQuestion):
    references:list[str]=Field(description="Citation Motivating your new answer")