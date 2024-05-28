from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableConfig
from agent import Assistant
from datetime import datetime

from database import Database
from typing import List, Union
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
from langchain_core.tools import BaseTool



# initialize tools and LLM
llm = ChatCohere()
embedding = CohereEmbeddings()
database = Database(data_dir="storage/database")
policy = Policy(data_dir="storage/policy", llm=llm, embedding=embedding)
flight_manager = FlightManager(database, llm)
car_manager = CarManager(database, llm)
hotel_manager = HotelManager(database, llm) 
excursions_manager = ExcursionsManager(database, llm)

flight_tools = flight_manager.get_tools()
search_flights = flight_tools['search_flights_tool']
user_flight = flight_tools['fetch_user_flight_information_tool']
update_ticket_to_new_flight = flight_tools['update_ticket_to_new_flight_tool']
cancel_ticket = flight_tools['cancel_ticket_tool']


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
    

    
def get_model_description(model: BaseModel) -> str:
    # Extract the class name
    model_name = model.__name__
    
    # Extract the class description (docstring)
    model_description = model.__doc__ or 'No description provided'
    
    # Extract the fields' information
    fields_info = [
        f"{name}: {field_info.type_} ({field_info.field_info.description or 'No description'})"
        for name, field_info in model.__fields__.items()
    ]
    fields_info_string = ', '.join(fields_info)
    
    # Format the result string
    result = (
        f"model_name -> {model_name}\n"
        f"model_params -> {fields_info_string}\n"
        f"model_description ->\n{model_description}"
    )
    
    return result

def get_description(tools: List[Union[BaseTool, BaseModel]]) -> str:
    models_list = []
    tools_list = []
    for i in tools:
        try:
            i.name
            tools_list.append(i) 
        except:
            models_list.append(i)           
    baseModel_desc = '\n\n'.join([get_model_description(tool) for tool in models_list])
    baseTool_desc = '\n\n'.join([get_tool_description(tool) for tool in tools_list])
    return baseModel_desc + baseTool_desc


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }


# Flight booking assistant

flight_booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling flight updates. "
            " The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that a booking isn't completed until after the relevant tool has successfully been used."
            """You have access to the following tools to get more information if needed:

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
            Do NOT add anything before or after the json response."""
            "\n\nCurrent user flight information:\n\n{user_info}\n"
            "\nCurrent time: {time}."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then"
            ' "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.',
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

update_flight_safe_tools = [search_flights, user_flight]
update_flight_sensitive_tools = [update_ticket_to_new_flight, cancel_ticket]
update_flight_tools = update_flight_safe_tools + update_flight_sensitive_tools
flight_tools_all = update_flight_tools + [CompleteOrEscalate]
update_flight_runnable = flight_booking_prompt.partial(tool_descs = get_description(flight_tools_all)) | llm

# # Hotel Booking Assistant
# book_hotel_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a specialized assistant for handling hotel bookings. "
#             "The primary assistant delegates work to you whenever the user needs help booking a hotel. "
#             "Search for available hotels based on the user's preferences and confirm the booking details with the customer. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
#             " Remember that a booking isn't completed until after the relevant tool has successfully been used."
#             "\nCurrent time: {time}."
#             '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant.'
#             " Do not waste the user's time. Do not make up invalid tools or functions."
#             "\n\nSome examples for which you should CompleteOrEscalate:\n"
#             " - 'what's the weather like this time of year?'\n"
#             " - 'nevermind i think I'll book separately'\n"
#             " - 'i need to figure out transportation while i'm there'\n"
#             " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
#             " - 'Hotel booking confirmed'",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())

# book_hotel_safe_tools = [search_hotels]
# book_hotel_sensitive_tools = [book_hotel, update_hotel, cancel_hotel]
# book_hotel_tools = book_hotel_safe_tools + book_hotel_sensitive_tools
# book_hotel_runnable = book_hotel_prompt | llm.bind_tools(
#     book_hotel_tools + [CompleteOrEscalate]
# )

# # Car Rental Assistant
# book_car_rental_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a specialized assistant for handling car rental bookings. "
#             "The primary assistant delegates work to you whenever the user needs help booking a car rental. "
#             "Search for available car rentals based on the user's preferences and confirm the booking details with the customer. "
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
#             " Remember that a booking isn't completed until after the relevant tool has successfully been used."
#             "\nCurrent time: {time}."
#             "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
#             '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
#             "\n\nSome examples for which you should CompleteOrEscalate:\n"
#             " - 'what's the weather like this time of year?'\n"
#             " - 'What flights are available?'\n"
#             " - 'nevermind i think I'll book separately'\n"
#             " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
#             " - 'Car rental booking confirmed'",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())

# book_car_rental_safe_tools = [search_car_rentals]
# book_car_rental_sensitive_tools = [
#     book_car_rental,
#     update_car_rental,
#     cancel_car_rental,
# ]
# book_car_rental_tools = book_car_rental_safe_tools + book_car_rental_sensitive_tools
# book_car_rental_runnable = book_car_rental_prompt | llm.bind_tools(
#     book_car_rental_tools + [CompleteOrEscalate]
# )

# # Excursion Assistant

# book_excursion_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a specialized assistant for handling trip recommendations. "
#             "The primary assistant delegates work to you whenever the user needs help booking a recommended trip. "
#             "Search for available trip recommendations based on the user's preferences and confirm the booking details with the customer. "
#             "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
#             " When searching, be persistent. Expand your query bounds if the first search returns no results. "
#             " Remember that a booking isn't completed until after the relevant tool has successfully been used."
#             "\nCurrent time: {time}."
#             '\n\nIf the user needs help, and none of your tools are appropriate for it, then "CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
#             "\n\nSome examples for which you should CompleteOrEscalate:\n"
#             " - 'nevermind i think I'll book separately'\n"
#             " - 'i need to figure out transportation while i'm there'\n"
#             " - 'Oh wait i haven't booked my flight yet i'll do that first'\n"
#             " - 'Excursion booking confirmed!'",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# ).partial(time=datetime.now())

# book_excursion_safe_tools = [search_trip_recommendations]
# book_excursion_sensitive_tools = [book_excursion, update_excursion, cancel_excursion]
# book_excursion_tools = book_excursion_safe_tools + book_excursion_sensitive_tools
# book_excursion_runnable = book_excursion_prompt | llm.bind_tools(
#     book_excursion_tools + [CompleteOrEscalate]
# )


# Primary Assistant
class ToFlightBookingAssistant(BaseModel):
    """Transfers work to a specialized assistant to handle flight updates and cancellations and Search for flights."""

    request: str = Field(
        description="Any necessary followup questions the update flight assistant should clarify before proceeding."
    
    )


# class ToBookCarRental(BaseModel):
#     """Transfers work to a specialized assistant to handle car rental bookings."""

#     location: str = Field(
#         description="The location where the user wants to rent a car."
#     )
#     start_date: str = Field(description="The start date of the car rental.")
#     end_date: str = Field(description="The end date of the car rental.")
#     request: str = Field(
#         description="Any additional information or requests from the user regarding the car rental."
#     )

#     class Config:
#         schema_extra = {
#             "example": {
#                 "location": "Basel",
#                 "start_date": "2023-07-01",
#                 "end_date": "2023-07-05",
#                 "request": "I need a compact car with automatic transmission.",
#             }
#         }


# class ToHotelBookingAssistant(BaseModel):
#     """Transfer work to a specialized assistant to handle hotel bookings."""

#     location: str = Field(
#         description="The location where the user wants to book a hotel."
#     )
#     checkin_date: str = Field(description="The check-in date for the hotel.")
#     checkout_date: str = Field(description="The check-out date for the hotel.")
#     request: str = Field(
#         description="Any additional information or requests from the user regarding the hotel booking."
#     )

#     class Config:
#         schema_extra = {
#             "example": {
#                 "location": "Zurich",
#                 "checkin_date": "2023-08-15",
#                 "checkout_date": "2023-08-20",
#                 "request": "I prefer a hotel near the city center with a room that has a view.",
#             }
#         }


# class ToBookExcursion(BaseModel):
#     """Transfers work to a specialized assistant to handle trip recommendation and other excursion bookings."""

#     location: str = Field(
#         description="The location where the user wants to book a recommended trip."
#     )
#     request: str = Field(
#         description="Any additional information or requests from the user regarding the trip recommendation."
#     )

#     class Config:
#         schema_extra = {
#             "example": {
#                 "location": "Lucerne",
#                 "request": "The user is interested in outdoor activities and scenic views.",
#             }
#         }


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            """You have access to the following tools to get more information if needed:

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
            Do NOT add anything before or after the json response."""
            "\n\nCurrent user flight information:\n\n{user_info}\n"
            "\nCurrent time: {time}."),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())
primary_assistant_tools = [
    PersianTavilySearchTool(max_results=20, llm=llm),
    # search_flights,
    policy.get_tools()['lookup_policy_tool'],
]

primary_tools = primary_assistant_tools + [
        ToFlightBookingAssistant,
        # ToBookCarRental,
        # ToHotelBookingAssistant,
        # ToBookExcursion,
    ]
assistant_runnable = primary_assistant_prompt.partial(tool_descs=get_description(primary_tools)) | llm