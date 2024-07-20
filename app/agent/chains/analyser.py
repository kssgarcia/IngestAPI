from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


# Prompt
analysis = PromptTemplate(
    template="""You a nutritionist data analizer and you always complete your taks the best you can\n 
    use the information of the user to describe his actual nutrition state. make enphasis in the user objectives to create your desription\n
    here is an example: \n

    First, let's go over some basic details you've shared with us. You are 35 years old, and you lead a sedentary lifestyle working in an office. Your physical measurements include a height of 1.75 meters, a weight of 90 kilograms, a BMI of 29.4, and a waist circumference of 102 centimeters. Your biochemical indicators show elevated levels of glucose and cholesterol. Your diet indicates a preference for processed and fast food, with low consumption of fruits, vegetables, and fiber, and high intake of saturated fats and sugars. Today, for instance, you had cereal with milk, a burger with fries, and pizza. Socially, you are single, have a medium income, and fortunately, you have access to healthy foods. Your physical activity level is low, with daily activities including office work, watching television, and reading, resulting in an energy consumption of 2016 kilocalories. Your goals are to reduce weight and waist circumference, improve your glucose and cholesterol levels, increase intake of fruits, vegetables, and fiber, reduce intake of processed foods, saturated fats, and sugars, increase your physical activity, improve cardiovascular health, and reduce diabetes risks. Potential health concerns identified are type 2 diabetes, heart diseases, hypertension, and being overweight or obese.\n
    Here is the data you're going to analize: \n\n {data}.\n Please only return analisis no preamble or explanation of what you have done. if the the data seem's to be empty just say tahta there were not user's information. your response mut to be in spanish \n """,
    input_variables=["data"],
)
# LLM
def analyser(local_llm:str, llm:ChatOllama, analysis:PromptTemplate=analysis):
    # llm = ChatOllama(model=local_llm, temperature=0, num_ctx=8000)

    analyser = analysis | llm | StrOutputParser()

    return analyser