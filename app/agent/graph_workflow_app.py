from langgraph.graph import END, StateGraph
from .graph_state import GraphState

from .flow_state import *
from pprint import pprint

# Define the nodes
def setup_lang_app():
    workflow = StateGraph(GraphState) 
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()

    return app

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
