from typing import List, Dict, Annotated, Any, Optional, Sequence, Literal
from typing_extensions import TypedDict

import uuid
import json
import warnings
from datetime import datetime

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt import ToolNode

from langgraph.utils import RunnableCallable
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import tools_condition
from langgraph.graph.message import add_messages

from database import Database
from policy import Policy
from online_search import PersianTavilySearchTool
from flight import FlightManager
from CarRental import CarManager
from Hotel import HotelManager
from Excursion import ExcursionsManager
from llm_translation import translate_to_persian
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Callable

    
    
def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "update_flight",
                "book_car_rental",
                "book_hotel",
                "book_excursion",
            ]
        ],
        update_dialog_stack,
    ]

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node
class Assistant:
    def __init__(self, runnable: Runnable, tools: List[BaseTool]):
        self.runnable = runnable
        self.tools = tools

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            print(state)
            if isinstance(state['messages'][-1], ToolMessage):
                if not state['messages'][-1].content:
                    state['messages'][-1].content = 'Checked parameters of called tool again and return a Json Blob with correct and alternative parameters \
                        in "ACTION_PARAMS"'
            
            result = self.runnable.invoke(state, config)
            try:
                if result.content == '':
                    # no answer from model
                    warnings.warn('WRONG EMPTY RESPOND: ' + result.content)
                    final_answer = ""
                    break
                    
                parser = JsonOutputParser()                    
                content_json = parser.invoke(result.content)
            except ValueError as e:
                warnings.warn('BAD FORMAT: \n' + result.content)
                state['messages'] += [result, HumanMessage("Respond with a valid json output!")]
                continue
            
            if isinstance(content_json, list):
                content_json = content_json[0]
            action = content_json.get('ACTION', '').replace(' ', '')
            action_params = content_json.get('ACTION_PARAMS') or {}
            if type(action_params) is str:
                action_params = json.loads(action_params)
            final_answer = content_json.get('FINAL_ANSWER')
            toolsModelInstance_list = []
            toolsToolInstance_list = []
            for i in self.tools:
                try:
                    i.name
                    toolsToolInstance_list.append(i.name)
                except:
                    toolsModelInstance_list.append(i.__name__)
                    
            tools_name = toolsModelInstance_list + toolsToolInstance_list
            print(tools_name)
            if action and action not in tools_name:
                warnings.warn('BAD TOOL NAME: ' + result.content)
                state['messages'] += [result, HumanMessage(f"The ACTION `{action}` does not exist!")]
                continue
            break

        if action and not final_answer:
            tool_call = ToolCall(name=action, args=action_params, id=str(uuid.uuid4()))
            result.tool_calls.append(tool_call)
            return {'messages': result}
        
        if not final_answer:
            persian_final_answer = "مدل پاسخی ندارد"
        else:
            pass
            # persian_final_answer = translate_to_persian(final_answer, self.runnable)

        # final_result = AIMessage(persian_final_answer)
        if not isinstance(final_answer, str):
            final_answer = str(final_answer)
        final_result = AIMessage(final_answer)

        return {'messages': final_result}
    
def user_info(state: State):
    llm = ChatCohere()
    database = Database(data_dir="storage/database")
    flight_data = FlightManager(db=database, llm=llm)
    fetch_user_info_tool = flight_data.get_tools().get('fetch_user_flight_information_tool')
    data = fetch_user_info_tool.invoke({})
    return {"user_info": data}


class Agent:

    def __init__(self) -> None:
        self.database = Database(data_dir="storage/database")
        self._printed_messages = set()

    def _print_event(
        self, event: dict, printed_messages: set,
    ) -> None:
        current_state = event.get('dialog_state')
        if current_state:
            print(f"Currently in: ", current_state[-1])

        messages = event.get('messages')
        if messages:
            for message in messages:
                if message.id not in printed_messages:
                    msg_repr = message.pretty_repr(html=True)
                    print(msg_repr)
                    printed_messages.add(message.id)


    def run(
        self, question: str, config: Dict, _graph: CompiledGraph,
        reset_db: bool = True, clear_message_history: bool = True,
    ) -> None:
        if reset_db:
            self.database.reset_and_prepare()

        new_messages = []
        if clear_message_history:
            _graph.checkpointer.put(config, checkpoint=empty_checkpoint())
        user_message = HumanMessage(question)
        new_messages.append(user_message)

        events = _graph.stream(
            {'messages': new_messages, 'user_info': None}, config, stream_mode='values'
        )

        for event in events:
            self._print_event(event, self._printed_messages)
        
        snapshot = _graph.get_state(config)
        while snapshot.next:
            
            # user_input = input(
            # "Do you approve of the above actions? Type 'y' to continue;"
            # " otherwise, explain your requested changed.\n\n"
            #                         )
            user_input = "i dont want to use that tool"
            y = "y"
            print('user answer is Y')
            # if user_input.strip() == "y":
            if y == "y":
                result = _graph.invoke(
                    None,
                    config,
                )                   
                self._print_event(result, self._printed_messages)
            else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
                result = _graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                    content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )
                self._print_event(result, self._printed_messages)   
                
                
            snapshot = _graph.get_state(config)
