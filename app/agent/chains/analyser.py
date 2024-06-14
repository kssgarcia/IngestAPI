from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# LLM
def analyser(local_llm:str):
    llm = ChatOllama(model=local_llm, temperature=0)


    # Prompt
    analysis = PromptTemplate(
        template="""You a nutritionist data analizer and you always complete your taks the best you can\n 
        use the information of the user to describe his actual nutrition state. make enphasis in the user objectives to create your desription\n
        here is an example: \n
        This data model details patient information for creating medical diagnoses. It includes user information with the email "sappps33ado@gmail.com". The dietary and health section covers age (35 years), sedentary lifestyle, and office job. In anthropometry, height (1.75 meters), weight (90 kilograms), BMI (29.4), and waist circumference (102 centimeters) are recorded. Biochemical indicators show elevated glucose and cholesterol levels. Regarding diet, the patient prefers processed food and fast food, with low consumption of fruits, vegetables, and fiber, and high consumption of saturated fats and sugars, with daily meals such as cereal with milk, burger with fries, and pizza. Social indicators indicate that the patient is single, has medium income, and access to healthy foods. Physical activity is low, with daily activities including office work, watching television, and reading, consuming 2016 kilocalories in these activities. The patient's goals are to reduce weight and waist circumference, improve glucose and cholesterol levels, increase intake of fruits, vegetables, and fiber, reduce intake of processed foods, saturated fats, and sugars, increase physical activity, improve cardiovascular health, and reduce diabetes risks. Finally, potential diseases identified are type 2 diabetes, heart diseases, hypertension, and overweight/obesity.\n
        Here is the data you're going to analize: \n\n {data}.\n Please only return analisis no preamble or explanation of what you have done. if the the data seem's to be empty just say tahta there were not user's information. your response mut to be in spanish \n """,
        input_variables=["data"],
    )

    analyser = analysis | llm | StrOutputParser()

    return analyser