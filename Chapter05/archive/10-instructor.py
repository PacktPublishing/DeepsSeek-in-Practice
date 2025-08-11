# requires-python = ">=3.12"
# /// script
# dependencies = [
#     "instructor",
# ]
# ///


import os
from openai import OpenAI
import instructor

from pydantic import BaseModel, Field

client = instructor.from_openai(
    OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"),
    mode=instructor.Mode.MD_JSON,  # because this is a reasoning model..
)


class City(BaseModel):
    name: str = Field(description="The name of the city")
    population: int = Field(description="The population of the city")
    notable_landmarks: list[str] = Field(
        description="List of notable landmarks in the city"
    )


class CitiesResponse(BaseModel):
    cities: list[City] = Field(description="List of interesting cities")


cities: CitiesResponse = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {
            "role": "user",
            "content": "Top 3 most interesting cities in Portugal",
        },
    ],
    response_model=CitiesResponse,
    temperature=0.0,
)

for city in cities.cities:
    print(f"City: {city.name}")
    print(f"Population: {city.population}")
    print(f"Notable Landmarks: {', '.join(city.notable_landmarks)}")
