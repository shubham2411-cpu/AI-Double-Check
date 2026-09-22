from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()


def verify(question, solver_answer):
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=f"""
You are the Verifier in an AI verification system.

Your job is to independently check the Solver's answer.

Check:
1. Is the answer correct?
2. Is the reasoning correct?
3. Are there calculation errors?
4. Are there missing assumptions or important details?

Do NOT blindly trust the Solver.

User question:
{question}

Solver's answer:
{solver_answer}

Give your verdict as either PASS or FAIL, followed by a short explanation.
"""
    )

    return response.output_text


question = input("Enter the original question: ")
solver_answer = input("Enter the Solver's answer: ")

result = verify(question, solver_answer)

print("\n--- VERIFIER RESULT ---")
print(result)
