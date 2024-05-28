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

class HotelManager:
    
    def __init__(self, db: Database, llm: BaseChatModel) -> None:
        self.db = db
        self.llm = llm
        
    def search_hotels(
        self,
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
    ) -> list[dict]:
        """
        Search for hotels based on location, name, price tier, check-in date, and check-out date.

        Args:
            location (Optional[str]): The location of the hotel. Defaults to None.
            name (Optional[str]): The name of the hotel. Defaults to None.
            price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
            checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

        Returns:
            list[dict]: A list of hotel dictionaries matching the search criteria.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        query = "SELECT * FROM hotels WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For the sake of this tutorial, we will let you match on any dates and price tier.
        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        return results


    def book_hotel(self, hotel_id: int) -> str:
        """
        Book a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to book.

        Returns:
            str: A message indicating whether the hotel was successfully booked or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute("UPDATE hotels SET booked = 1 WHERE id = ?", (hotel_id,))
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Hotel {hotel_id} successfully booked."
        else:
            connection.close()
            return f"No hotel found with ID {hotel_id}."



    def update_hotel(
        self,
        hotel_id: int,
        checkin_date: Optional[Union[datetime, date]] = None,
        checkout_date: Optional[Union[datetime, date]] = None,
    ) -> str:
        """
        Update a hotel's check-in and check-out dates by its ID.

        Args:
            hotel_id (int): The ID of the hotel to update.
            checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

        Returns:
            str: A message indicating whether the hotel was successfully updated or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        if checkin_date:
            cursor.execute(
                "UPDATE hotels SET checkin_date = ? WHERE id = ?", (checkin_date, hotel_id)
            )
        if checkout_date:
            cursor.execute(
                "UPDATE hotels SET checkout_date = ? WHERE id = ?",
                (checkout_date, hotel_id),
            )

        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Hotel {hotel_id} successfully updated."
        else:
            connection.close()
            return f"No hotel found with ID {hotel_id}."


    def cancel_hotel(self, hotel_id: int) -> str:
        """
        Cancel a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to cancel.

        Returns:
            str: A message indicating whether the hotel was successfully cancelled or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute("UPDATE hotels SET booked = 0 WHERE id = ?", (hotel_id,))
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Hotel {hotel_id} successfully cancelled."
        else:
            connection.close()
            return f"No hotel found with ID {hotel_id}."

        
    def get_tools(self) -> Dict[str, BaseTool]:
        tools = [
            search_hotels_Tool(hotel_manager=self),
            book_hotel_Tool(hotel_manager=self),
            update_hotel_Tool(hotel_manager=self),
            cancel_hotel_Tool(hotel_manager=self),
        ]
        return {tool.name: tool for tool in tools}
    

class search_hotels_Input(BaseModel):
    location: Optional[str] = Field(description='location (Optional[str]): The location of the hotel. Defaults to None.')
    name: Optional[str] = Field(description='name (Optional[str]): The name of the hotel. Defaults to None.')
    price_tier: Optional[str] = Field(
        description='price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury'
        )
    checkin_date: Optional[Union[datetime, date]] = Field(
        description='checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.'
        )
    checkout_date: Optional[Union[datetime, date]] = Field(
        description='checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.'
        )


class search_hotels_Tool(BaseTool):

    name = 'search_hotels_tool'
    description = (
        """
        Search for hotels based on location, name, price tier, check-in date, and check-out date.

        Args:
            location (Optional[str]): The location of the hotel. Defaults to None.
            name (Optional[str]): The name of the hotel. Defaults to None.
            price_tier (Optional[str]): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
            checkin_date (Optional[Union[datetime, date]]): The check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The check-out date of the hotel. Defaults to None.

        Returns:
            list[dict]: A list of hotel dictionaries matching the search criteria.
        """
    )
    args_schema: Type[BaseModel] = search_hotels_Input
    return_direct: bool = False

    hotel_manager: HotelManager

    def _run(
        self, 
        location: Optional[str],
        name: Optional[str],
        price_tier: Optional[str],
        checkin_date: Optional[Union[datetime, date]],
        checkout_date: Optional[Union[datetime, date]],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.hotel_manager.search_hotels(
            location, name, price_tier, checkin_date, checkout_date
        )
        return list_of_dict_to_str(results)
    

class book_hotel_Input(BaseModel):
    hotel_id: Optional[int] = Field(description='hotel_id (int): The ID of the hotel to update.')


class book_hotel_Tool(BaseTool):

    name = 'book_hotel_tool'
    description = (
        """
        Book a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to book.

        Returns:
            str: A message indicating whether the hotel was successfully booked or not.
        """
    )
    args_schema: Type[BaseModel] = book_hotel_Input
    return_direct: bool = False

    hotel_manager: HotelManager

    def _run(
        self, 
        hotel_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.hotel_manager.book_hotel(
            hotel_id
        )
        return results

        
class update_hotel_Input(BaseModel):
    hotel_id: Optional[int] = Field(description='hotel_id (int): The ID of the hotel to update.')
    checkin_date: Optional[Union[datetime, date]] = Field(
        description='checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.'
        )
    checkout_date: Optional[Union[datetime, date]] = Field(
        description='checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.'
        )


class update_hotel_Tool(BaseTool):

    name = 'update_hotel_tool'
    description = (
        """
        Update a hotel's check-in and check-out dates by its ID.

        Args:
            hotel_id (int): The ID of the hotel to update.
            checkin_date (Optional[Union[datetime, date]]): The new check-in date of the hotel. Defaults to None.
            checkout_date (Optional[Union[datetime, date]]): The new check-out date of the hotel. Defaults to None.

        Returns:
            str: A message indicating whether the hotel was successfully updated or not.
        """
    )
    args_schema: Type[BaseModel] = update_hotel_Input
    return_direct: bool = False

    hotel_manager: HotelManager

    def _run(
        self, 
        hotel_id: Optional[int],
        checkin_date: Optional[Union[datetime, date]],
        checkout_date: Optional[Union[datetime, date]],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.hotel_manager.update_hotel(
            hotel_id, checkin_date, checkout_date
        )
        return results
    

class cancel_hotel_Input(BaseModel):
    hotel_id: Optional[int] = Field(description='hotel_id (int): The ID of the hotel to cancel.')


class cancel_hotel_Tool(BaseTool):

    name = 'cancel_hotel_tool'
    description = (
        """
        Cancel a hotel by its ID.

        Args:
            hotel_id (int): The ID of the hotel to cancel.

        Returns:
            str: A message indicating whether the hotel was successfully cancelled or not.
        """
    )
    args_schema: Type[BaseModel] = cancel_hotel_Input
    return_direct: bool = False

    hotel_manager: HotelManager

    def _run(
        self, 
        hotel_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.hotel_manager.cancel_hotel(
            hotel_id
        )
        return results
