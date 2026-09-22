from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import run_pipeline


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Double-Check",
    description="Multi-agent AI answer verification system",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Double-Check API is running."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# VERIFY QUESTION
# ============================================================

@app.post("/verify")
def verify_question(request: QuestionRequest):

    result = run_pipeline(request.question)

    return result