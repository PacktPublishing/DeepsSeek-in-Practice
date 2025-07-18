# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
#     "openai",
# ]
# ///

import asyncio
from openai import OpenAI
import os
import json
from fastmcp import Client

# MCP Server Configuration
mcp_server_config = {
    "mcpServers": {"assistant": {"command": "uv", "args": ["run", "./mcp_server.py"]}}
}


# Helper functions to get tools and call them
async def fetch_async_tools() -> list:
    async with Client(mcp_server_config) as mcp_client:
        return await mcp_client.list_tools()


def get_tools() -> list:
    """Fetch and format the list of tools from the MCP server."""
    mcp_tools = asyncio.run(fetch_async_tools())
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


def call_tool(tool_name: str, arguments: dict) -> str:
    """Call a tool with the given name and arguments."""

    async def a_call_tool(tool_name, arguments):
        async with Client(mcp_server_config) as mcp_client:
            return await mcp_client.call_tool(tool_name, arguments)

    tool_result = asyncio.run(a_call_tool(tool_name, arguments))
    tool_result = tool_result.content[0].text

    return tool_result


# Initialize OpenAI client with DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
tools = get_tools()


def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

tool_result = call_tool(tool.function.name, json.loads(tool.function.arguments))
print(f"Tool>\t {tool.function.name}({tool.function.arguments})")
print(f"Result>\t {tool_result}")

messages.append(
    {
        "role": "tool",
        "tool_call_id": tool.id,
        "content": tool_result,
    }
)
message = send_messages(messages)
print(f"Model>\t {message.content}")

