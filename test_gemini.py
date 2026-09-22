from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

response = client.interactions.create(
    model="gemini-3.8-flash",
    input="Say hello in one short sentence."
)

print(response.output_text)