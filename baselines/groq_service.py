import os
import aiohttp
import asyncio

async def summarize_with_groq(text: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text."
            },
            {
                "role": "user",
                "content": f"Please summarize the following text:\n\n{text}"
            }
        ],
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        # Respect rate limits, e.g., 30 requests per minute
        await asyncio.sleep(2)
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result['choices'][0]['message']['content']
            else:
                response.raise_for_status()