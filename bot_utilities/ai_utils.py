import aiohttp
import base64
import io
import json
from datetime import datetime
import re
import asyncio
import time
from bot_utilities.config_loader import load_current_language, config
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

openai_client = AsyncOpenAI(
    api_key=os.getenv('XAI_API_KEY'),
    base_url="https://api.x.ai/v1"
)

# Tool definitions for the Responses API
XAI_TOOLS = [
    {
        "type": "web_search",
    },
    {
        "type":"file_search",
        "vector_store_ids": ["collection_354bc49e-8970-4876-bd97-47a4c1683933","collection_3c22fb13-1340-40a9-8b47-4f6a0bb92c0e"],
        "max_num_results":20
    },
    {
        "type":"mcp",
        "server_url":"https://rhen.tyler.ad/mcp",
        "server_label":"google-workspace"
    }
]


def _extract_text(response) -> str | None:
    """Extract the first output_text from a Responses API response."""
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    return content.text
    return None


async def search(query):
    """Search the web via DuckDuckGo and return results as a formatted string."""
    if not query or len(query) > 200:
        return "No search results found."

    search_results_limit = config['MAX_SEARCH_RESULTS']
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blob = f"Search results for: '{query}' at {current_time}:\n"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://ddg-api.awam.repl.co/api/search',
                params={'query': query, 'maxNumResults': search_results_limit}
            ) as response:
                results = await response.json()
    except aiohttp.ClientError as e:
        print(f"Search request error: {e}")
        return "Search failed."

    for index, result in enumerate(results):
        try:
            blob += f'[{index}] "{result["Snippet"]}"\n\nURL: {result["Link"]}\n'
        except Exception as e:
            blob += f'Search error: {e}\n'

    blob += "\nSearch results provide real-time information. Include relevant links in your response if helpful.\n"
    return blob


async def generate_response(instructions, history):
    """
    Generate a response using the xAI Responses API with automatic tool calling.
    The model autonomously calls search_web when real-time information is needed.
    """
    input_messages = [
        {"role": "system", "content": instructions},
        *history,
    ]
    tools = XAI_TOOLS if internet_access else []

    response = await openai_client.responses.create(
        model=config['GPT_MODEL'],
        input=input_messages,
        tools=tools,
    )

    # Tool calling loop — cap at 5 rounds to prevent runaway calls
    for _ in range(5):
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            break

        tool_outputs = []
        for item in tool_calls:
            if item.name == "search_web":
                args = json.loads(item.arguments)
                result = await search(args.get("query", ""))
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })

        response = await openai_client.responses.create(
            model=config['GPT_MODEL'],
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools,
        )

    return _extract_text(response)


async def generate_grok_response(prompt):
    """One-shot Responses API call with no history or tools."""
    response = await openai_client.responses.create(
        model=config['GPT_MODEL'],
        input=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response)


async def poly_image_gen(prompt):
    response = await openai_client.images.generate(
        model="grok-imagine-image",
        prompt=prompt,
        n=1,
        response_format="b64_json",
    )
    image_data = base64.b64decode(response.data[0].b64_json)
    return io.BytesIO(image_data)


async def generate_image_prodia(prompt, model, sampler, seed, neg):
    print("\033[1;32m(xAI) Creating image for :\033[0m", prompt)
    start_time = time.time()

    full_prompt = prompt
    if neg:
        full_prompt = f"{prompt}. Avoid: {neg}"

    response = await openai_client.images.generate(
        model="grok-imagine-image",
        prompt=full_prompt,
        n=1,
        response_format="b64_json",
    )
    image_data = base64.b64decode(response.data[0].b64_json)
    img_file_obj = io.BytesIO(image_data)
    duration = time.time() - start_time
    print(f"\033[1;34m(xAI) Finished image creation\n\033[0mPrompt: {prompt} in {duration:.2f} seconds.")
    return img_file_obj
