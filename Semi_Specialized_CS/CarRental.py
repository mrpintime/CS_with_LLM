from datetime import date, datetime
from typing import Optional, Union
from typing import List, Dict, Type, Optional

import pytz

from langchain_core.runnables import ensure_config
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from database import Database
from utils import list_of_dict_to_str

#TODO: Complete Description of each fields in  tools input

class CarManager:
    
    def __init__(self, db: Database, llm: BaseChatModel) -> None:
        self.db = db
        self.llm = llm
        
    def search_car_rentals(self,
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
    ) -> list[dict]:
        """
        Search for car rentals based on location, name, price tier, start date, and end date.

        Args:
            location (Optional[str]): The location of the car rental. Defaults to None.
            name (Optional[str]): The name of the car rental company. Defaults to None.
            price_tier (Optional[str]): The price tier of the car rental. Defaults to None.
            start_date (Optional[Union[datetime, date]]): The start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The end date of the car rental. Defaults to None.

        Returns:
            list[dict]: A list of car rental dictionaries matching the search criteria.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        query = "SELECT * FROM car_rentals WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        # For our tutorial, we will let you match on any dates and price tier.
        # (since our toy dataset doesn't have much data)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        results = [dict(zip(column_names, row)) for row in rows]

        cursor.close()
        connection.close()
        
        return results


    def book_car_rental(self, rental_id: int) -> str:
        """
        Book a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to book.

        Returns:
            str: A message indicating whether the car rental was successfully booked or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 1 WHERE id = ?", (rental_id,))
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Car rental {rental_id} successfully booked."
        else:
            connection.close()
            return f"No car rental found with ID {rental_id}."



    def update_car_rental(self,
        rental_id: int,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
    ) -> str:
        """
        Update a car rental's start and end dates by its ID.

        Args:
            rental_id (int): The ID of the car rental to update.
            start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

        Returns:
            str: A message indicating whether the car rental was successfully updated or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        if start_date:
            cursor.execute(
                "UPDATE car_rentals SET start_date = ? WHERE id = ?",
                (start_date, rental_id),
            )
        if end_date:
            cursor.execute(
                "UPDATE car_rentals SET end_date = ? WHERE id = ?", (end_date, rental_id)
            )

        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Car rental {rental_id} successfully updated."
        else:
            connection.close()
            return f"No car rental found with ID {rental_id}."


    def cancel_car_rental(self, rental_id: int) -> str:
        """
        Cancel a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to cancel.

        Returns:
            str: A message indicating whether the car rental was successfully cancelled or not.
        """
        connection = self.db.get_connection()
        cursor = connection.cursor()

        cursor.execute("UPDATE car_rentals SET booked = 0 WHERE id = ?", (rental_id,))
        connection.commit()

        if cursor.rowcount > 0:
            connection.close()
            return f"Car rental {rental_id} successfully cancelled."
        else:
            connection.close()
            return f"No car rental found with ID {rental_id}."
        
    def get_tools(self) -> Dict[str, BaseTool]:
        tools = [
            search_car_rentals_Tool(car_manager=self),
            book_car_rental_Tool(car_manager=self),
            update_car_rental_Tool(car_manager=self),
            cancel_car_rental_Tool(car_manager=self),
        ]
        return {tool.name: tool for tool in tools}
    
    

class search_car_rentals_Input(BaseModel):
    location: Optional[str] = Field(description='This have to be Full Name of The city of the car rental')
    name: Optional[str] = Field(description='The name of the car rental company.')
    price_tier: Optional[str] = Field(description='The price tier of the car rental.')
    start_date: Optional[Union[datetime, date]] = Field(description='start_date (Optional[Union[datetime, date]]): The start date of the car rental.')
    end_date: Optional[Union[datetime, date]] = Field(description='end_date (Optional[Union[datetime, date]]): The end date of the car rental.')
    
class search_car_rentals_Tool(BaseTool):

    name = 'search_car_rentals_tool'
    description = (
        """
        Search for car rentals based on location, name, price tier, start date, and end date.
        """
    )
    args_schema: Type[BaseModel] = search_car_rentals_Input
    return_direct: bool = False

    car_manager: CarManager

    def _run(
        self, 
        location: Optional[str] = None,
        name: Optional[str] = None,
        price_tier: Optional[str] = None,
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime , date]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        config = ensure_config()
        configuration = config.get('configurable', {})
        passenger_id = configuration.get('passenger_id', None)
        if not passenger_id:
            raise ValueError("No `passenger_id` configured.")

        results = self.car_manager.search_car_rentals(
            location, name, price_tier, start_date, end_date
        )
        return list_of_dict_to_str(results)
    
    
class book_car_rental_Input(BaseModel):
    rental_id: Optional[int] = Field(description='rental_id (int): The ID of the car rental to book.')


class book_car_rental_Tool(BaseTool):

    name = 'book_car_rental_tool'
    description = (
        """
        Book a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to book.

        Returns:
            str: A message indicating whether the car rental was successfully booked or not.
        """
    )
    args_schema: Type[BaseModel] = book_car_rental_Input
    return_direct: bool = False

    car_manager: CarManager

    def _run(
        self, 
        rental_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.car_manager.book_car_rental(
            rental_id
        )
        return results
    
    
class update_car_rental_Input(BaseModel):
    rental_id: Optional[int] = Field(description='rental_id (int): The ID of the car rental to update.')
    start_date: Optional[Union[datetime, date]] = Field(description='start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.')
    end_date: Optional[Union[datetime, date]] = Field(description='end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.')


class update_car_rental_Tool(BaseTool):

    name = 'update_car_rental_tool'
    description = (
        """
        Update a car rental's start and end dates by its ID.

        Args:
            rental_id (int): The ID of the car rental to update.
            start_date (Optional[Union[datetime, date]]): The new start date of the car rental. Defaults to None.
            end_date (Optional[Union[datetime, date]]): The new end date of the car rental. Defaults to None.

        Returns:
            str: A message indicating whether the car rental was successfully updated or not.
        """
    )
    args_schema: Type[BaseModel] = update_car_rental_Input
    return_direct: bool = False

    car_manager: CarManager

    def _run(
        self, 
        rental_id: Optional[int],
        start_date: Optional[Union[datetime, date]] = None,
        end_date: Optional[Union[datetime, date]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.car_manager.update_car_rental(
            rental_id, start_date, end_date
        )
        return results
    


class cancel_car_rental_Input(BaseModel):
    rental_id: Optional[int] = Field(description='rental_id (int): The ID of the car rental to cancel.')


class cancel_car_rental_Tool(BaseTool):

    name = 'cancel_car_rental_tool'
    description = (
        """
        Cancel a car rental by its ID.

        Args:
            rental_id (int): The ID of the car rental to cancel.

        Returns:
            str: A message indicating whether the car rental was successfully cancelled or not.
        """
    )
    args_schema: Type[BaseModel] = cancel_car_rental_Input
    return_direct: bool = False

    car_manager: CarManager

    def _run(
        self, 
        rental_id: Optional[int],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        results = self.car_manager.update_car_rental(
            rental_id
        )
        return results
