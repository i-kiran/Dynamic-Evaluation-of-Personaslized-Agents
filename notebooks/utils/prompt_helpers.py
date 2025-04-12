def generate_prompt_for_search(profile: dict, interview_text: str) -> str:
    """
    Generates a prompt to query an LLM for a product search phrase based on user profile and interview text.
    
    Returns:
        A string that can be sent to the LLM with clear instructions for JSON output containing keys:
        - search_phrase: a concise, natural language search query suitable for product databases.
        - explanation: a brief rationale explaining how the phrase was generated.
    """
    
    prompt = f"""
You are a helpful assistant that specializes in personalized e-commerce recommendations.

Your task is to generate a **natural-sounding search phrase** that can be used to query a product database. 
Use the information provided in:
1. The **user profile**, which contains demographics, preferences, brand affinities, and values.
2. The **interview text**, which is a conversation about specific shopping needs and preferences.

Desirable chararectistics:
- Keep the phrase **short and realistic** — aim for **under 5 words**, like how humans usually search.
- Use **intent from the conversation** and **preferences from the profile**.
- DO NOT generate full sentences or overly descriptive queries.


Your response must be in **strict JSON format**, with the following structure:
{{
  "search_phrase": "<insert concise natural language query here>",
  "explanation": "<insert short explanation of how you used profile and conversation>"
}}

Here is the user profile:
{profile}

Here is the interview text:
\"\"\"{interview_text}\"\"\"

Only return the JSON. Do not include any other text.
The search phrase should be optimized for search engines or internal product databases (e.g., Amazon).
Make sure to reflect relevant brands, product categories, design/style preferences, and functional requirements.

Use your best judgment to synthesize key themes into the search phrase.

"""
    return prompt
