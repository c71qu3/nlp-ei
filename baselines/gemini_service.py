import os
import aiohttp
import asyncio

async def summarize_with_gemini(text: str, api_key: str) -> str:
    if not api_key:
        raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {
        "x-goog-api-key": f"{api_key}",
        "Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{
                "text": f"Please summarize the following text:\n\n{text}"
            }]
        }]
    }

    async with aiohttp.ClientSession() as session:
        # Respect rate limits, e.g., 60 requests per minute
        await asyncio.sleep(1)
        async with session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                response.raise_for_status()
