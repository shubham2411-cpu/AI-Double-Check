from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import run_pipeline


app = FastAPI(
    title="AI Double-Check",
    description="Multi-agent AI answer verification system",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


# Serve frontend
app.mount(
    "/app",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)


@app.get("/")
def home():
    return {
        "message": "AI Double-Check API is running.",
        "frontend": "/app"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify")
def verify_question(request: QuestionRequest):
    try:
        result = run_pipeline(request.question)
        return result

    except Exception as e:
        error_message = str(e)

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message.upper()
            or "rate limit" in error_message.lower()
        ):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "gemini_rate_limit_exceeded",
                    "detail": (
                        "Gemini API rate limit reached (HTTP 429): "
                        "the free-tier daily quota is exhausted. "
                        "Try again later, after the quota resets, "
                        "or use an API key with a higher-tier plan."
                    )
                }
            )

        raise