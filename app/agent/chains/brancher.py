##Decider
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Prompt
analysis = PromptTemplate(
    template="""You a sentence analizer assistant. analize the sentence blow and decide whether this sentence needs nutritional information or not. based on what you say another resource is going to be use to give the nutritional information to the user, analyse the sentence well for the resource to not be wasted.\n sentence: {sentence} \n 
    if the sentence needs nutritional information to be answerd you gotta grade it as 'yes'
    if not you gotta grade it as 'no'.
    your answer is either 'yes' or 'no' and the output  structure should be like this: 'nutrition':'yes'""",
    input_variables=["sentence"],
)

def branchDecider(local_llm:str, llm_json:ChatOllama, analysis:PromptTemplate=analysis, ):
    llm_json.format="json"
    decider = analysis | llm_json | JsonOutputParser()

    return decider