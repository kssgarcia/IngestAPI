#state flow definition
from langchain.schema import Document
from .chains.generator_chain import generator
from .chains.generator_chain import commonGenerator
from .chains.retriever_grader_chain import retrieval_grader
from .chains.rewriter_chain import rewriter
from .chains.analyser import analyser
from .chains.brancher import branchDecider
from .retriever import *

#state
from .graph_state import GraphState, Query

#web tool library
# Web search
from langchain_community.retrievers import TavilySearchAPIRetriever

#memory libraries
from langchain_community.chat_message_histories import ChatMessageHistory



local_llm="llama3"



#Document loader
docuemt=document_loader(filename="TEMASDENUTRICINBSICABook.md")
#splitter
all_splits=text_splitter(docuemt)
#Embedding and vector store
vectorstore = store_and_retrieve(all_splits)

#memory
chat_history = ChatMessageHistory()
 
#tools
web_search_tool = TavilySearchAPIRetriever(k=3)
retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#Chains
generate_with_summarization=generator(local_llm=local_llm,chat_history=chat_history)
retrieval_grader=retrieval_grader(local_llm=local_llm)
question_rewriter=rewriter(local_llm=local_llm)
analyser=analyser(local_llm=local_llm)
branchDecider=branchDecider(local_llm=local_llm)
commonGenerator=commonGenerator(local_llm=local_llm, chat_history=chat_history)





def branch(state):
    """
    Decides wether documents are needed or not based on the question

    Args:
    state (Document): The current state of the document

    Returns:
    state (Document): The new state with a new nuutritionalBranch key
    """
    print("---BRANCH---")

    question=state["question"]
    userdata= state["userdata"]
    diagnosis = state["diagnosis"]
    print("estoy aqui")
    required = branchDecider.invoke({"sentence":question})
    return {"question":question, "userdata":userdata, "nutritionBranch":required }

def nutritionRequired(state):
    """
    Decides whether the given question requires nutritional information

    Args:
        state (dict): The current graph state

    Returns:ß
        Binary decision, generate or go into the nutritional branch
    """
    print("---Nutrition Required---")
    question = state["question"]
    userdata= state["userdata"]
    required=state["nutritionBranch"]

    # Retrieval
    
    if required["nutrition"]=='no':
        print("---NO NUTRITION REQUIRED---")
        return "generateCommon"
    
    if required["nutrition"]=='yes':
        print("---NUTRITION REQUIRED---")
        return "retrieve"

def generateCommon(state):
    """
    Generate common answers to the question

    Args:
    state (dict): The current graph state

    Returns:
    state(dict): The updated graph state with a new generation key
    """
    question = state["question"]
    userdata=state["userdata"]
    print(question)
    
    print("---GENERATE COMMON---")

    commonGeneration=commonGenerator.invoke(input={"question":question}, config={"configurable": {"session_id": "unused"}})
    return {"question":question, "generation":commonGeneration}




def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:ß
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    userdata=state["userdata"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def formatuserdata(state):
    """ 
    Analyze user data and format it for it to be used in the model.

    Args:
     state (dict): The current graph state
    
    Returns:
        state (dict): New key added to state, userdata, that contains formatted user data
    """
    print("---FORMATUSERDATA---")
    user_data = state["userdata"]
    documents = state["documents"]
    question = state["question"]


    # Format user data
    formatted_data = (
        f"Email: {user_data.userData.email}, "
        f"Age: {user_data.dietetics.age}, "
        f"Lifestyle: {user_data.dietetics.lifestyle}, "
        f"Job: {user_data.dietetics.job}, "
        f"Anthropometry: (Height: {user_data.dietetics.anthropometry.height}, "
        f"Weight: {user_data.dietetics.anthropometry.weight}, "
        f"BMI: {user_data.dietetics.anthropometry.BMI}, "
        f"Waist Circumference: {user_data.dietetics.anthropometry.waist_circumference}), "
        f"Biochemical Indicators: (Glucose: {user_data.dietetics.biochemical_indicators.glucose}, "
        f"Cholesterol: {user_data.dietetics.biochemical_indicators.cholesterol}), "
        f"Diet: (Ingest Preferences: {', '.join(user_data.dietetics.diet.ingest_preferences)}, "
        f"Fruits and Vegetables: {user_data.dietetics.diet.fruits_and_vegetables}, "
        f"Fiber: {user_data.dietetics.diet.fiber}, "
        f"Saturated Fats: {user_data.dietetics.diet.saturated_fats}, "
        f"Sugars: {user_data.dietetics.diet.sugars}, "
        f"Today Meals: {', '.join(user_data.dietetics.diet.today_meals)}), "
        f"Social Indicators: (Marital Status: {user_data.dietetics.social_indicators.marital_status}, "
        f"Income: {user_data.dietetics.social_indicators.income}, "
        f"Access to Healthy Foods: {user_data.dietetics.social_indicators.access_to_healthy_foods}), "
        f"Physical Activity: {user_data.dietetics.physical_activity}, "
        f"Daily Activities: {', '.join(user_data.dietetics.daily_activities)}, "
        f"Activities Energy Consumption: {user_data.dietetics.activities_energy_consumption}, "
        f"Goals: (Reduce Weight: {user_data.dietetics.goals.reduce_weight}, "
        f"Reduce Waist Circumference: {user_data.dietetics.goals.reduce_waist_circumference}, "
        f"Improve Glucose Cholesterol Levels: {user_data.dietetics.goals.improve_glucose_cholesterol_levels}, "
        f"Increase Intake Fruits Vegetables Fiber: {user_data.dietetics.goals.increase_intake_fruits_vegetables_fiber}, "
        f"Reduce Intake Processed Food Saturated Fats Sugars: {user_data.dietetics.goals.reduce_intake_processed_food_saturated_fats_sugars}, "
        f"Increase Physical Activity: {user_data.dietetics.goals.increase_physical_activity}, "
        f"Improve Cardio Health: {user_data.dietetics.goals.improve_cardio_health}, "
        f"Reduce Diabetes Risks: {user_data.dietetics.goals.reduce_diabetes_risks}), "
        f"Potential Diseases: (Type 2 Diabetes: {user_data.dietetics.potential_diseases.type_2_diabetes}, "
        f"Heart Diseases: {user_data.dietetics.potential_diseases.heart_diseases}, "
        f"Hypertension: {user_data.dietetics.potential_diseases.hypertension}, "
        f"Overweight Obesity: {user_data.dietetics.potential_diseases.overweight_obesity})"
    )

    user_data = analyser.invoke(input={formatted_data})
    return {"userdata": user_data, "question": question, "documents": documents, "generation":user_data}



def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    user_data = state["userdata"]
    print(question, documents)

    # RAG generation
    generation= generate_with_summarization.invoke(
    {"question": question, "context":documents, "user_data":user_data},
    {"configurable": {"session_id": "unused"}},)
    # generation = rag_chain.invoke({"context": documents, "question": question, "messages":chat_history.messages})
    # chat_history.add_ai_message(generation)
    # print(generation)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        print(d)
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    print (question)
    print(documents)

    # Web search
    from langchain_community.retrievers import TavilySearchAPIRetriever

    web_search_tool = TavilySearchAPIRetriever(k=3)
    docs = web_search_tool.invoke(question)
    documents.append(docs)
    print("---WEB SEARCH RESULTS---")
    print(docs)

    return {"documents": documents, "question": question}


### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"