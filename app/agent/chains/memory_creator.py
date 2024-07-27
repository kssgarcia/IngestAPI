from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama


#Summary from user message
memory_prompt = PromptTemplate(

    template="""You are a nutritionist assistant and your main task is to create memories of what the patients say. Include as many specific details as you can. Here is an example of what you do:\n 
    patient statement: "Hello my name is Yilber"
    the momory you return: "memories":"The patient name is Yilber", "mainProblem":"The patient name is Yilber"
    \n 
    here is another EXAMPLE case in which a list of memories is necessary:\n 
    patient: “Hola, doctora. Estoy un poco preocupado por mi alimentación. Siento que no estoy haciendo las cosas bien y quiero mejorar mi salud. ¿Podría ayudarme? Normalmente desayuno un café con galletas o pan. A veces me salto el almuerzo porque estoy muy ocupado en el trabajo. En la cena, suelo comer bastante, especialmente carnes y papas fritas. También me encanta picar entre comidas, especialmente dulces.”\n 
    the memory you returned:
    'memories': [
        "El paciente está preocupado por su alimentación y desea mejorar su salud.",
        "El paciente desayuna normalmente un café con galletas o pan.",
        "El paciente a veces se salta el almuerzo debido a su ocupación en el trabajo.",
        "El paciente cena bastante, especialmente carnes y papas fritas.",
        "El paciente disfruta picar entre comidas, especialmente dulces."
    ],
    'mainProblem': "El paciente está preocupado por su alimentación y desea mejorar su salud."\n 
    
        
    It is always needed that you subdivide what the patient says in many ideas as you can identify. Moreover, the patient's ideas tend to be realted to a main problem, which is important that you are able to identify it. the answer should be returned in a json format, with this structure:
    "memories":["el paciente...", "el paciente..."], "mainProblem":"el paciente..."

    Complete your task for this case:
    {statement}\n your ansewr should only contain the json structure: \n 'memories':["el paciente...", "el paciente..."], 'mainProblem':"el paciente..."\n no premable text or conlusions. only the raw json structure containing the answer you considered. """,        
        input_variables=["statement"] 
    )

def memo_creator(local_llm:str,llm_json:ChatOllama, memory_prompt:PromptTemplate=memory_prompt, ):

    #memories chain
    memo_creator=memory_prompt | llm_json | JsonOutputParser()
    
    return memo_creator