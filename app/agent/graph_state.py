from typing_extensions import TypedDict
from typing import List, Annotated
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from PIL import Image

#state flow definition
from langchain.schema import Document

#Graph Creation
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        user_data: user data
    """

    question: str
    generation: Annotated[list, add_messages]
    web_search: str
    documents: List[str]
    userdata: dict = {}
    nutritionBranch:str
    diagnosis:str='no'
    sessionid:str=''
    image=Image

class Query(BaseModel):
    question: str