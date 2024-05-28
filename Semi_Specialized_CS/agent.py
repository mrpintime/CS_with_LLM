from typing import List, Dict, Annotated, Any, Optional, Sequence, Literal
from typing_extensions import TypedDict

import uuid
import json
import warnings
from datetime import datetime

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


def get_tool_description(tool: BaseTool) -> str:
    tool_params = [
        f"{name}: {info.get('type', ' ')} ({info.get('description', ' ')})"
        for name, info in tool.args.items()
    ]
    tool_params_string = ', '.join(tool_params)
    return (
        f"tool_name -> {tool.name}\n"
        f"tool_params -> {tool_params_string}\n"
        f"tool_description ->\n{tool.description}"
    )


def get_tools_description(tools: List[BaseTool]) -> str:
    return '\n\n'.join([get_tool_description(tool) for tool in tools])


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable, tools: List[BaseTool]):
        self.runnable = runnable
        self.tools = tools

    def __call__(self, state: State, config: RunnableConfig):
        while True:
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
                print(content_json)
            action = content_json.get('ACTION', '').replace(' ', '')
            action_params = content_json.get('ACTION_PARAMS') or {}
            if type(action_params) is str:
                action_params = json.loads(action_params)
            final_answer = content_json.get('FINAL_ANSWER')
            
            
            if action and action not in [tool.name for tool in self.tools]:
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
            persian_final_answer = translate_to_persian(final_answer, self.runnable)

        final_result = AIMessage(persian_final_answer)

        return {'messages': [result, final_result]}
    
def user_info(state: State):
    llm = ChatCohere()
    database = Database(data_dir="storage/database")
    flight_data = FlightManager(db=database, llm=llm)
    fetch_user_info_tool = flight_data.get_tools().get('fetch_user_flight_information_tool')
    data = fetch_user_info_tool.invoke({})
    return {"user_info": data}


SYSTEM_PROMPT_TEMPLATE = \
"""
You are a helpful Persian customer support assistant for Iran Airlines.
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If a search comes up empty, expand your search before giving up.

You have access to the following tools to get more information if needed:

{tool_descs}

You also have access to the history of previous messages.
Return your response as a JSON blob with keys as a dict type.

If you have any questions from the user, put that in `FINAL_ANSWER` as well.

JSON blob when you need to use a tool MUST have ONLY following keys:

"THOUGHT": "<you should always think about what to do>",
"ACTION": "<the action to take, must be one tool_name from above tools>",
"ACTION_PARAMS": "<the input parameters to the ACTION, it must be in json format complying with the tool_params>"

JSON blob when you do not need to use a tool MUST have ONLY following keys:
"THOUGHT": "<you should always think about what to do>",
"FINAL_ANSWER": "<a text containing the final answer to the original input question>",


Always you have to look at the previous messages and try to find answer from them and if you can find answer of user's question from previous messages then put answer in "FINAL_ANSWER".

Always make sure that your output is a JSON blob complying with above format.
Do NOT add anything before or after the json response.

Current user:\n<User>\n{user_info}\n</User>
Current time: {time}.
"""

class Agent:

    def __init__(self) -> None:
        # new_address = "https://31eb-34-91-57-236.ngrok-free.app"
        self.assistant_prompt = ChatPromptTemplate.from_messages( [("system",SYSTEM_PROMPT_TEMPLATE), ("placeholder", "{messages}") ])
        self.llm = ChatCohere()
        self.embedding = CohereEmbeddings()
        self.database = Database(data_dir="storage/database")
        self.policy = Policy(data_dir="storage/policy", llm=self.llm, embedding=self.embedding)
        self.flight_manager = FlightManager(self.database, self.llm)
        self.car_manager = CarManager(self.database, self.llm)
        self.hotel_manager = HotelManager(self.database, self.llm) 
        self.excursions_manager = ExcursionsManager(self.database, self.llm)
        
        self.sensitive_tools = [
            self.flight_manager.get_tools()['update_ticket_to_new_flight_tool'],
            self.flight_manager.get_tools()['cancel_ticket_tool'],
            self.car_manager.get_tools()['book_car_rental_tool'],
            self.car_manager.get_tools()['update_car_rental_tool'],
            self.car_manager.get_tools()['cancel_car_rental_tool'],
            self.hotel_manager.get_tools()['book_hotel_tool'],
            self.hotel_manager.get_tools()['update_hotel_tool'],
            self.hotel_manager.get_tools()['cancel_hotel_tool'],
            self.excursions_manager.get_tools()['book_excursion_tool'],
            self.excursions_manager.get_tools()['update_excursion_tool'],
            self.excursions_manager.get_tools()['cancel_excursion_tool']
        ]
        self.sensitive_tools_names = [tool.name for tool in self.sensitive_tools]
        
        self.safe_tools = [
            PersianTavilySearchTool(max_results=20, llm=self.llm),
            self.flight_manager.get_tools()['fetch_user_flight_information_tool'],
            self.flight_manager.get_tools()['search_flights_tool'],
            self.policy.get_tools()['lookup_policy_tool'],
            self.car_manager.get_tools()['search_car_rentals_tool'],
            self.hotel_manager.get_tools()['search_hotels_tool'],                
            self.excursions_manager.get_tools()['search_trip_recommendations_tool']
        ]
        self.safe_tools_names = [tool.name for tool in self.safe_tools]

        self._graph = self._build_graph()
        self._printed_messages = set()
        


    @property
    def tools(self) -> List[BaseTool]:
        policy_tools = list(self.policy.get_tools().values())
        flight_tools = list(self.flight_manager.get_tools().values())
        car_rental_tools = list(self.car_manager.get_tools().values())
        hotels_tools = list(self.hotel_manager.get_tools().values())
        excursions_tools = list(self.excursions_manager.get_tools().values())
        search_tool = [PersianTavilySearchTool(max_results=20, llm=self.llm)]

        return search_tool + policy_tools + flight_tools + car_rental_tools + excursions_tools + hotels_tools

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
        

    def _create_tool_node_with_fallback(self, tools: List[BaseTool]) -> Dict:
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self._handle_tool_error)], exception_key='error'
        )
    
    def _route_tools(self, state: State) -> Literal["safe_tools", "sensitive_tools", "__end__"]:
        next_node = tools_condition(state)
        # If no tools are invoked, return to the user
        if next_node == END:
            return END
        ai_message = state["messages"][-1]
        # This assumes single tool calls. To handle parallel tool calling, you'd want to
        # use an ANY condition
        first_tool_call = ai_message.tool_calls[0]
        if first_tool_call["name"] in self.sensitive_tools_names:
            return "sensitive_tools"
        return "safe_tools"

    def _build_graph(self) -> CompiledGraph:
        builder = StateGraph(State)
        self.llm_assistant =  self.assistant_prompt.partial(time=datetime.now(), tool_descs=get_tools_description(self.tools)) | self.llm
        builder.add_node("fetch_user_info", user_info)
        builder.set_entry_point('fetch_user_info')
        builder.add_edge("fetch_user_info", "assistant")
        builder.add_node('assistant', Assistant(self.llm_assistant, self.tools))
        builder.add_node("safe_tools", self._create_tool_node_with_fallback(self.safe_tools))
        builder.add_node("sensitive_tools", self._create_tool_node_with_fallback(self.sensitive_tools))
        # builder.add_node('action', self._create_tool_node_with_fallback(self.tools))
        # builder.add_conditional_edges(
        #     'assistant',
        #     tools_condition
        # )
        
        builder.add_conditional_edges(
            "assistant",
            self._route_tools,
        )
        builder.add_edge("safe_tools", "assistant")
        builder.add_edge("sensitive_tools", "assistant")
        
        memory = SqliteSaver.from_conn_string(':memory:')
        graph = builder.compile(
            checkpointer=memory,interrupt_before=["sensitive_tools"] )
            # The graph will always halt before executing the "tools" node.
            # The user can approve or reject (or even alter the request) before
            # the assistant continues
        return graph

    def _print_event(
        self, event: dict, printed_messages: set,
        ignore_first_system_message: bool = False,
    ) -> None:
        current_state = event.get('dialog_state')
        if current_state:
            print(f"Currently in: ", current_state[-1])

        messages = event.get('messages')
        if messages:
            if ignore_first_system_message:
                if isinstance(messages[0], SystemMessage):
                    messages = messages[1:]

            for message in messages:
                if message.id not in printed_messages:
                    msg_repr = message.pretty_repr(html=True)
                    print(msg_repr)
                    printed_messages.add(message.id)


    def run(
        self, question: str, config: Dict,
        reset_db: bool = True, clear_message_history: bool = True,
    ) -> None:
        if reset_db:
            self.database.reset_and_prepare()

        new_messages = []
        if clear_message_history:
            self._graph.checkpointer.put(config, checkpoint=empty_checkpoint())
        user_message = HumanMessage(question)
        new_messages.append(user_message)

        events = self._graph.stream(
            {'messages': new_messages, 'user_info': None}, config, stream_mode='values'
        )

        for event in events:
            self._print_event(event, self._printed_messages)
        
        snapshot = self._graph.get_state(config)
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
                result = self._graph.invoke(
                    None,
                    config,
                )                   
                self._print_event(result, self._printed_messages)
            else:
            # Satisfy the tool invocation by
            # providing instructions on the requested changes / change of mind
                result = self._graph.invoke(
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
                
                
            snapshot = self._graph.get_state(config)
