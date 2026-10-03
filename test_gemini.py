from dotenv import load_dotenv
from google import genai


def test_gemini_connection():
    load_dotenv()

    client = genai.Client()

    response = client.interactions.create(
        model="gemini-3.8-flash",
        input="Say hello in one short sentence."
    )

    print("Gemini connection successful.")
    print("Response:", response.output_text)


if __name__ == "__main__":
    test_gemini_connection()