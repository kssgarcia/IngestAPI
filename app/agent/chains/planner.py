from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

prompt = """
For the following task, create detailed plans to answer the question step by step. Ensure each step logically follows from the previous one. For each plan, specify the external tool to use along with the tool input to retrieve evidence. Store the evidence in a variable #E that can be referenced by later steps. (Plan, #E1, Plan, #E2, Plan, ...)

Available tools:
1. Google[input]: A worker that searches for results on Google. Useful for finding short, succinct answers on specific topics. The input should be a search query.
2. LLM[input]: A pretrained LLM like yourself. Useful for tasks requiring general world knowledge and common sense. Additionally, the LLM has the role of a nutritionist, capable of answering questions that require nutritional knowledge. When a Google search result relates to nutrition, the LLM should analyze it from the nutritionist perspective. The LLM can reference Google results by citing the relevant step (given #E: the number of the step containing relevant information). The input can be any instruction or question.

Example:
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours less than Toby. How many hours did Rebecca work?
Plan: Translate the problem into algebraic expressions and solve with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Determine the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details, but without providing explanations for each step. Each plan should be followed by only one #E. Avoid any preamble or conclusion; return only the raw plan. Each step can reference the results of previous steps by citing #E1, #E2, etc.

Task: {task}
"""


prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])

def plannerchain(local_llm:str,prompt_template:ChatPromptTemplate=prompt_template):
    llm = ChatOllama(model=local_llm, temperature=0)
    planner = prompt_template | llm
    return planner