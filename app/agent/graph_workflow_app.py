from langgraph.graph import START, END, StateGraph
from .graph_state import GraphState

from .flow_state import *
from pprint import pprint

# Define the nodes
def setup_lang_app():
    workflow2 = StateGraph(GraphState)
    # Define the nodes
    workflow2.add_node("initialize", inicialize)  # inialize
    workflow2.add_node("branch", branch)# start node
    workflow2.add_node("generateCommon", generateCommon)  #generate common kind answers
    workflow2.add_node("retrieve", retrieve)  # retrieve documents
    # workflow2.add_node("analysis", formatuserdata)  # userdata
    workflow2.add_node("grade_documents", grade_documents)  # grade documents
    workflow2.add_node("generate", generate)  # generatae
    workflow2.add_node("recentmessages", recent_messages_add)#create a lil memo if needed
    workflow2.add_node("plan", get_plan) # make a plan to answer
    workflow2.add_node("tool", tool_execution) # execute tools
    workflow2.add_node("solve", solve) #generate an asnwer based on the plan




    # Build graph
    workflow2.add_edge(START, "branch")
    workflow2.add_edge(START,"generateCommon")
    workflow2.add_edge(START,"retrieve")

    #------------Coninue
    workflow2.add_edge(["branch","retrieve"],"initialize")
    workflow2.add_edge("initialize", "plan")
    workflow2.add_edge("initialize", "grade_documents")


    #--------generate common

    workflow2.add_edge(["generate","generateCommon"],"recentmessages")

    #--------- Retrive based on documents
    # workflow2.add_edge("retrieve", "analysis")
    # workflow2.add_edge("retrieve", "grade_documents")
    # workflow2.add_edge("grade_documents", "plan")
    workflow2.add_edge(["solve","grade_documents"], "generate")

    #--------- Conditional

    # workflow2.add_edge("grade_documents","generate")
    # workflow2.add_edge("generate", "recentmessages")


    #-------- Parallel plan
    workflow2.add_edge("plan", "tool")
    workflow2.add_conditional_edges("tool", _route, {"solve":"solve","tool":"tool"})
    # workflow2.add_edge("solve", "recentmessages")

    workflow2.add_edge("recentmessages", END)

    app2 = workflow2.compile()

    return app2
# langgraph_app=setup_lang_app()

# thread = {"configurable": {"thread_id": "4"}}
# result = []
# inputs = {"question": "nutricion, hablame"}
# for event in langgraph_app.stream(inputs, stream_mode="values"):
#     for key, value in event.items():
#         # Node
#         (f"Node '{key}':")
#         # Optional: print full state at each node
#         # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#     pprint("\n---\n")
# pprint(value["generation"])
# # except Exception as e:
# #     raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
