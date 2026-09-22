from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()


def solve(question):
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=f"""
You are the Solver in an AI verification system.

Solve the user's question carefully and logically.

User question:
{question}

Give a clear answer with reasoning where useful.
"""
    )

    return response.output_text


question = input("Enter your question: ")

answer = solve(question)

print("\n--- SOLVER ANSWER ---")
print(answer)