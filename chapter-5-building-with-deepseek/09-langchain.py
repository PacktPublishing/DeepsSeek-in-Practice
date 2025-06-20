# requires-python = ">=3.12"
# /// script
# dependencies = [
#     "langchain",
#     "langchain-deepseek",
#     "pydantic",
# ]
# ///


from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


class SkyColorEnum(str, Enum):
    BLUE = "blue"
    GRAY = "gray"
    WHITE = "white"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    CLEAR = "clear"


class SkyColor(BaseModel):
    """Sky color prediction for Copenhagen."""

    color: SkyColorEnum = Field(description="The primary color of the sky")
    description: str = Field(description="Detailed description of the sky appearance")
    confidence: Optional[int] = Field(
        default=None, description="Confidence level of the prediction, from 1 to 10"
    )


structured_llm = llm.with_structured_output(SkyColor)
result: SkyColor = structured_llm.invoke(
    "What color is the sky likely to be in Copenhagen today?"
)  # type: ignore

print(result.color)
