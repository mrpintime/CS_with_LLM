from typing import List, Dict, Annotated, Any, Optional, Sequence, Literal
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from agent import Assistant, State, user_info, create_entry_node
from Specialized_Assistants import (
    update_flight_runnable, update_flight_sensitive_tools, update_flight_safe_tools, 
    CompleteOrEscalate, assistant_runnable, primary_assistant_tools, ToFlightBookingAssistant, flight_tools_all, primary_tools
    )




builder = StateGraph(State)

def _handle_tool_error(self, state: State) -> Dict:
    error = state.get('error')
    tool_call = state['messages'][-1].tool_calls[0]
    return {
        'messages': ToolMessage(
            content=(
                f"Error: {repr(error)}\n please fix your mistakes."
            ),
            tool_call_id=tool_call['id'],
        )
    }
        

def _create_tool_node_with_fallback(tools: List[BaseTool]) -> Dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(_handle_tool_error)], exception_key='error'
        )



# Flight Assistant
builder.add_node("fetch_user_info", user_info)
builder.set_entry_point("fetch_user_info")

# Flight booking assistant
builder.add_node(
    "enter_update_flight",
    create_entry_node("Flight Updates & Booking Assistant", "update_flight"),
)
builder.add_node("update_flight", Assistant(update_flight_runnable, tools=flight_tools_all))
builder.add_edge("enter_update_flight", "update_flight")
builder.add_node(
    "update_flight_sensitive_tools",
    _create_tool_node_with_fallback(tools=update_flight_sensitive_tools),
)
builder.add_node(
    "update_flight_safe_tools",
    _create_tool_node_with_fallback(tools=update_flight_safe_tools),
)


def route_update_flight(
    state: State,
) -> Literal[
    "update_flight_sensitive_tools",
    "update_flight_safe_tools",
    "leave_skill",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_toolnames = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_toolnames for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"


builder.add_edge("update_flight_sensitive_tools", "update_flight")
builder.add_edge("update_flight_safe_tools", "update_flight")
builder.add_conditional_edges("update_flight", route_update_flight)


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder.add_node("leave_skill", pop_dialog_state)
builder.add_edge("leave_skill", "primary_assistant")



# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable, tools=primary_tools))
builder.add_node(
    "primary_assistant_tools", _create_tool_node_with_fallback(primary_assistant_tools)
)


def route_primary_assistant(
    state: State,
) -> Literal[
    "primary_assistant_tools",
    "enter_update_flight",
    # "enter_book_hotel",
    # "enter_book_excursion",
    "__end__",
]:
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        # elif tool_calls[0]["name"] == ToBookCarRental.__name__:
        #     return "enter_book_car_rental"
        # elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
        #     return "enter_book_hotel"
        # elif tool_calls[0]["name"] == ToBookExcursion.__name__:
        #     return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


# The assistant can route to one of the delegated assistants,
# directly use a tool, or directly respond to the user
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    {
        "enter_update_flight": "enter_update_flight",
        "primary_assistant_tools": "primary_assistant_tools",
        # "enter_book_car_rental": "enter_book_car_rental",
        # "enter_book_hotel": "enter_book_hotel",
        # "enter_book_excursion": "enter_book_excursion",
        
        END: END,
    },
)
builder.add_edge("primary_assistant_tools", "primary_assistant")


# Each delegated workflow can directly respond to the user
# When the user responds, we want to return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    # "book_car_rental",
    # "book_hotel",
    # "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]


builder.add_conditional_edges("fetch_user_info", route_to_workflow)

# Compile graph
memory = SqliteSaver.from_conn_string(":memory:")
final_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "update_flight_sensitive_tools",
        # "book_car_rental_sensitive_tools",
        # "book_hotel_sensitive_tools",
        # "book_excursion_sensitive_tools",
    ],
)