from datetime import date, datetime
from typing import Optional, Union
from typing import List, Dict, Type, Optional


from langchain_core.runnables import ensure_config
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from database import Database
from utils import list_of_dict_to_str

#TODO: Complete Description of each fields in  tools input
#TODO: also we can change input of each tools to better understanding from LLM

class ExcursionsManager:
    
    def __init__(self, db: Database, llm: BaseChatModel) -> None:
        self.db = db
        self.llm = llm
        
    def search_trip_recommendations(
        self,
        location: Optional[str] = None,
        name: Optional[str] = None,
        keywords: Optional[str] = None,
    ) -> list[dict]:
        """
        Search for trip recommendations based on location, name, and keywords.

        Args:
            location (Optional[str]): The location of the trip recommendation. Defaults to None.
            name (Optional[str]): The name of the trip recommendation. Defaults to None.
            keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

        Returns:
            list[dict]: A list of trip recommendation dictionaries matching the search criteria.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()


        query = "SELECT * FROM trip_recommendations WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        if keywords:
            keyword_list = keywords.split(",")
            keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
            query += f" AND ({keyword_conditions})"
            params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        return results
    
    def book_excursion(self, recommendation_id: int) -> str:
        """
        Book a excursion by its recommendation ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to book.

        Returns:
            str: A message indicating whether the trip recommendation was successfully booked or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
        )
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Trip recommendation {recommendation_id} successfully booked."
        else:
            connection.close()
            return f"No trip recommendation found with ID {recommendation_id}."



    def update_excursion(self, recommendation_id: int, details: str) -> str:
        """
        Update a trip recommendation's details by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to update.
            details (str): The new details of the trip recommendation.

        Returns:
            str: A message indicating whether the trip recommendation was successfully updated or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET details = ? WHERE id = ?",
            (details, recommendation_id),
        )
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Trip recommendation {recommendation_id} successfully updated."
        else:
            connection.close()
            return f"No trip recommendation found with ID {recommendation_id}."


    def cancel_excursion(self, recommendation_id: int) -> str:
        """
        Cancel a trip recommendation by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to cancel.

        Returns:
            str: A message indicating whether the trip recommendation was successfully cancelled or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
        )
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Trip recommendation {recommendation_id} successfully cancelled."
        else:
            connection.close()
            return f"No trip recommendation found with ID {recommendation_id}."


        
    def get_tools(self) -> Dict[str, BaseTool]:
        tools = [
            search_trip_recommendations_Tool(excursions_manager=self),
            book_excursion_Tool(excursions_manager=self),
            update_excursion_Tool(excursions_manager=self),
            cancel_excursion_Tool(excursions_manager=self),
        ]
        return {tool.name: tool for tool in tools}


class search_trip_recommendations_Input(BaseModel):
    location: Optional[str] = Field(description='location (Optional[str]): The location of the trip recommendation. Defaults to None.')
    name: Optional[str] = Field(description='name (Optional[str]): The name of the trip recommendation. Defaults to None.')
    keywords: Optional[str] = Field(description='keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.')


class search_trip_recommendations_Tool(BaseTool):

    name = 'search_trip_recommendations_tool'
    description = (
        """
        Search for trip recommendations based on location, name, and keywords.

        Args:
            location (Optional[str]): The location of the trip recommendation. Defaults to None.
            name (Optional[str]): The name of the trip recommendation. Defaults to None.
            keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

        Returns:
            list[dict]: A list of trip recommendation dictionaries matching the search criteria.
        """
    )
    args_schema: Type[BaseModel] = search_trip_recommendations_Input
    return_direct: bool = False

    excursions_manager: ExcursionsManager

    def _run(
        self, 
        location: Optional[str],
        name: Optional[str],
        keywords: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.excursions_manager.search_trip_recommendations(
            location, name, keywords
        )
        return list_of_dict_to_str(results)
    

class book_excursion_Input(BaseModel):
    recommendation_id: Optional[int] = Field(description='recommendation_id (int): The ID of the trip recommendation to book.')


class book_excursion_Tool(BaseTool):

    name = 'book_excursion_tool'
    description = (
        """
        Book a excursion by its recommendation ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to book.

        Returns:
            str: A message indicating whether the trip recommendation was successfully booked or not.
        """
    )
    args_schema: Type[BaseModel] = book_excursion_Input
    return_direct: bool = False

    excursions_manager: ExcursionsManager

    def _run(
        self, 
        recommendation_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.excursions_manager.book_excursion(
            recommendation_id
        )
        return results

        
class update_excursion_Input(BaseModel):
    recommendation_id: Optional[int] = Field(description='recommendation_id (int): The ID of the trip recommendation to update.')
    details: Optional[str] = Field(description='details (str): The new details of the trip recommendation.')


class update_excursion_Tool(BaseTool):

    name = 'update_excursion_tool'
    description = (
        """
        Update a trip recommendation's details by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to update.
            details (str): The new details of the trip recommendation.

        Returns:
            str: A message indicating whether the trip recommendation was successfully updated or not.
        """
    )
    args_schema: Type[BaseModel] = update_excursion_Input
    return_direct: bool = False

    excursions_manager: ExcursionsManager

    def _run(
        self, 
        recommendation_id: Optional[int],
        details: Optional[str],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.excursions_manager.update_excursion(
            recommendation_id, details
        )
        return results
    

class cancel_excursion_Input(BaseModel):
    recommendation_id: Optional[int] = Field(description='recommendation_id (int): The ID of the trip recommendation to cancel.')


class cancel_excursion_Tool(BaseTool):

    name = 'cancel_excursion_tool'
    description = (
               """
        Cancel a trip recommendation by its ID.

        Args:
            recommendation_id (int): The ID of the trip recommendation to cancel.

        Returns:
            str: A message indicating whether the trip recommendation was successfully cancelled or not.
        """
    )
    args_schema: Type[BaseModel] = cancel_excursion_Input
    return_direct: bool = False

    excursions_manager: ExcursionsManager

    def _run(
        self, 
        recommendation_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.excursions_manager.cancel_excursion(
            recommendation_id
        )
        return results
