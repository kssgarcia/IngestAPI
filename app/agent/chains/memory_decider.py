from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama


#Summary from user message
memory_decider_prompt = PromptTemplate(

    template="""You are a nutritionist assistant. your unique task is to determine if a patient message contains valuable information of the patient or not. The valuable information can be anthropometric (height, weight, etc.), can be bichemical indicator, can be a dietetic (ingest related), can be social (marital status, incomem etc.) can be related to energy consumption (the activities the patient does during its day), can be realated to the patient nutrition goals (losing weight, improving cardio helth, etc), or can be also realted to the deseases the patient has (diabetes, hypertension, etc). 
    
    Here is an example of what you do:\n 
    patient statement: "Hello my name is Yilber"
    the momory you return: "memories":"The patient name is Yilber", "mainProblem":"The patient name is Yilber"\n 

    the message contains personal information about the patient, in this case its name. so you gotta return a binary answer saying 'yes' ir 'no' following this json srtructure:\n 'valuableinfo':'yes'
    \n 
    here is another example case in which a lot of info could be extracted:\n 
    patient: “Hola, doctora. Estoy un poco preocupado por mi alimentación. Siento que no estoy haciendo las cosas bien y quiero mejorar mi salud. ¿Podría ayudarme? Normalmente desayuno un café con galletas o pan. A veces me salto el almuerzo porque estoy muy ocupado en el trabajo. En la cena, suelo comer bastante, especialmente carnes y papas fritas. También me encanta picar entre comidas, especialmente dulces.”
    the memory you returned:
    "memories": [
        "El paciente está preocupado por su alimentación y desea mejorar su salud.",
        "El paciente desayuna normalmente un café con galletas o pan.",
        "El paciente a veces se salta el almuerzo debido a su ocupación en el trabajo.",
        "El paciente cena bastante, especialmente carnes y papas fritas.",
        "El paciente disfruta picar entre comidas, especialmente dulces."
    ],
    "mainProblem": "El paciente está preocupado por su alimentación y desea mejorar su salud."
    
    in this case there are several types of valuable information ranged from ingest related to dietetic related. so your answer should be 'yes' following this json structure:\n 'valuableinfo':'yes'

    here is an example of a message that does not contain valuable information:\n

    patient: "hola como estas, podrias ayudarme?"
    patient: "como esto ayuda a que nuestro cuerpo se repare?"
    patient: "cuentame mas de ti"
    patient: "que es lo que haces en tu vida"
    patient: "estoy haciendo muchas cosas para perder peso"
    partient: ""


    in this case, there is not so much information a nutritionist could use since what was said was general, not specific. so your answer should be 'no' following this json structure:\n 'valuableinfo':'no'\n It is important that you consider this, the information should be precise and really valuable. 

    your ansewr should only contain the json structure: \n 'valuableinfo':'yes' or 'valuableinfo':'no'\n no premable text or conlusions. only the raw json structure containing the answer you considered.

    here is the sentence you gotta analize:\n{patient_message}
    """,    
    input_variables=["patient_message"]    
    )


def memo_decider(local_llm:str, llm_json:ChatOllama, memory_decider_prompt:PromptTemplate=memory_decider_prompt):

    #memory decider
    memory_decider=memory_decider_prompt|llm_json | JsonOutputParser()
    return memory_decider