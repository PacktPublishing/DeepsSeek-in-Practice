# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
# ]
# ///

from fastmcp import FastMCP


# Create a server instance
mcp = FastMCP(name="MyAssistantServer")


@mcp.tool
def get_weather(city: str) -> str:
    """Multiplies two numbers."""
    return f"The weather in {city} is sunny with a high of 25°C."


@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    return {"name": f"User {user_id}", "status": "active"}


if __name__ == "__main__":
    mcp.run()
