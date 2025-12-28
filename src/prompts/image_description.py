"""
Image Description Prompts for Gemini Vision
"""

IMAGE_DESCRIPTION_PROMPT = """Analyze this surveillance footage and provide a concise description in MAXIMUM TWO LINES. 

IMPORTANT: Only describe what you CAN SEE. Do NOT mention what is absent or not visible.

Focus on:
1. People: Clothing colors and what they are doing (e.g., 'man in red shirt walking', 'woman in blue standing near gate')
2. Actions: Specific actions (e.g., 'entering property', 'walking', 'running', 'fighting')
3. Objects: Only mention clearly visible objects (vehicle, weapon, gate, door, etc.)
4. Context: Describe if actions look normal or aggressive

Distinguish between normal and threatening:
- Standing near someone or walking together = normal proximity
- Grabbing, hitting, or forcing someone = aggressive
- People in public spaces or near vehicles = normal activity
- Pointing weapon or attacking = threatening

Be factual and specific. Keep it to TWO LINES maximum.

Example: 'A man in a red jacket points a gun at a man in blue jeans near a white car. The man in blue has his hands raised and is backing away.'"""
