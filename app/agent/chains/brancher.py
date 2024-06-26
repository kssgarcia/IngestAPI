##Decider
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def branchDecider(llm:ChatOllama):
    analysis = PromptTemplate(
        template="""You a sentence analizer assistant. analize the sentence blow and decide whether this sentence needs nutritional information or not. based on what you say another resource is going to be use to give the nutritional information to the user, analyse the sentence well for the resource to not be wasted.\n sentence: {sentence} \n 
        your answer is either 'yes' or 'no' and the output  structure should be like this: 'nutrition':'yes'
        if the sentence needs nutritional information to be answerd you gotta respond with a 'nutrition':'yes' and
        if not you gotta respond it as 'nutrition':'no'.
        """,
        input_variables=["sentence"],
    )

    decider = analysis | llm | JsonOutputParser()

    return decider