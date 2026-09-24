from dotenv import load_dotenv
from google import genai
import time
import json
import re
import math
from dataclasses import dataclass


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = genai.Client()


# ============================================================
# STRUCTURED VERIFIER RESULT
# ============================================================

@dataclass
class VerifierResult:
    verdict: str          # "PASS" or "FAIL"
    confidence: float     # numeric, 0.0 .. 1.0
    explanation: str      # non-empty short rationale


# ============================================================
# GEMINI REQUEST HANDLER
# ============================================================

def ask_gemini(prompt):

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            return response.text

        except Exception as e:

            error_text = str(e)

            if "503" in error_text:

                if attempt < 2:

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying... ({attempt + 1}/3)"
                    )

                    time.sleep(2)

                    continue

            raise


# ============================================================
# SOLVER
# ============================================================

def solve(question):

    return ask_gemini(f"""
You are the Solver in an AI verification system.

Solve the user's question carefully.

Show your reasoning clearly.
State assumptions when necessary.
Check calculations, equations, units, and important details.

User question:
{question}

Provide a complete answer.
""")


# ============================================================
# VERIFIER
# ============================================================

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verifier_output(text):
    """
    Parse the Verifier's raw text into a VerifierResult.

    Expects a single JSON object with keys:
        verdict, confidence, explanation

    Raises ValueError with a descriptive message on any failure.
    """

    if text is None:

        raise ValueError(
            "Verifier returned no text."
        )

    match = _JSON_OBJECT_RE.search(text)

    if not match:

        raise ValueError(
            f"Verifier did not return a JSON object. Raw output: {text!r}"
        )

    raw_json = match.group(0)

    try:

        obj = json.loads(raw_json)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"Verifier output was not valid JSON: "
            f"{raw_json!r} ({e})"
        )

    if not isinstance(obj, dict):

        raise ValueError(
            f"Verifier output JSON must be an object, "
            f"got {type(obj).__name__}."
        )

    # ---- exact key set: verdict, confidence, explanation ----

    required_keys = (
        "verdict",
        "confidence",
        "explanation"
    )

    missing_keys = [
        k for k in required_keys
        if k not in obj
    ]

    if missing_keys:

        raise ValueError(
            f"Verifier output JSON is missing required key(s): "
            f"{missing_keys}."
        )

    extra_keys = [
        k for k in obj
        if k not in required_keys
    ]

    if extra_keys:

        raise ValueError(
            f"Verifier output JSON contains unexpected key(s): "
            f"{extra_keys}. Allowed keys are exactly: "
            f"{list(required_keys)}."
        )

    # ---- verdict: exactly PASS or FAIL ----

    raw_verdict = obj.get("verdict")

    if not isinstance(raw_verdict, str):

        raise ValueError(
            f"Verifier 'verdict' must be a string, got "
            f"{type(raw_verdict).__name__}: {raw_verdict!r}."
        )

    verdict = raw_verdict.strip().upper()

    if verdict not in ("PASS", "FAIL"):

        raise ValueError(
            f"Verifier 'verdict' must be exactly 'PASS' or 'FAIL', "
            f"got {raw_verdict!r}."
        )

    # ---- confidence: numeric, strictly in [0.0, 1.0] ----

    raw_confidence = obj.get("confidence")

    # bool is a subclass of int in Python; reject it explicitly.

    if isinstance(raw_confidence, bool) or not isinstance(
        raw_confidence,
        (int, float)
    ):

        raise ValueError(
            f"Verifier 'confidence' must be numeric, got "
            f"{type(raw_confidence).__name__}: {raw_confidence!r}."
        )

    confidence = float(raw_confidence)

    if not math.isfinite(confidence):

        raise ValueError(
            f"Verifier 'confidence' must be a finite number, "
            f"got {confidence}."
        )

    if confidence < 0.0 or confidence > 1.0:

        raise ValueError(
            f"Verifier 'confidence' must be between 0.0 and 1.0 "
            f"inclusive, got {confidence}."
        )

    # ---- explanation: non-empty string ----

    raw_explanation = obj.get("explanation")

    if not isinstance(raw_explanation, str):

        raise ValueError(
            f"Verifier 'explanation' must be a string, got "
            f"{type(raw_explanation).__name__}: {raw_explanation!r}."
        )

    explanation = raw_explanation.strip()

    if not explanation:

        raise ValueError(
            "Verifier 'explanation' must be a non-empty string."
        )

    return VerifierResult(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
    )


def verify(question, solver_answer):

    raw_output = ask_gemini(f"""
You are the Verifier in an AI verification system.

Independently check the Solver's answer.

Respond with a single JSON object and nothing else, using exactly
these three fields:

  "verdict":     the string "PASS" or the string "FAIL"
  "confidence":  a number between 0.0 and 1.0 (inclusive) expressing
                 how confident you are in the verdict
  "explanation": one short sentence (under 30 words) explaining the verdict

Do not include any prose, markdown, or code fences outside the JSON object.

User question:
{question}

Solver's answer:
{solver_answer}
""")

    try:

        return _parse_verifier_output(raw_output)

    except ValueError as e:

        raise RuntimeError(
            f"Failed to parse Verifier output into structured result: {e}"
        ) from e


# ============================================================
# CRITIC
# ============================================================

def _format_verifier(verification):

    """Render a VerifierResult as a labeled block for downstream prompts."""

    return (
        f"Verdict: {verification.verdict}\n"
        f"Confidence: {verification.confidence:.2f}\n"
        f"Explanation: {verification.explanation}"
    )


def criticize(question, solver_answer, verification):

    verification_text = _format_verifier(verification)

    return ask_gemini(f"""
You are the Critic in an AI verification system.

Review the Solver's answer and the Verifier's assessment.

Identify any remaining problems, missing reasoning, incorrect
assumptions, calculation errors, or weaknesses.

User question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification_text}

Give a concise critique.
""")


# ============================================================
# FINALIZER
# ============================================================

def finalize(
    question,
    solver_answer,
    verification,
    critique
):

    verification_text = _format_verifier(verification)

    return ask_gemini(f"""
You are the Finalizer in an AI verification system.

Produce the final answer to the user's question.

Use the Solver's answer as the starting point, but incorporate
the Verifier's assessment and the Critic's analysis.

Correct any errors identified during verification.

Do not mention the internal verification process unless necessary.

User question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification_text}

Critic's analysis:
{critique}

Return the final answer clearly and accurately.
""")


# ============================================================
# FULL GEMINI PIPELINE
# ============================================================

def run_gemini_pipeline(question):

    # Solver

    print("\n--- SOLVER ---")

    solver_answer = solve(question)

    print(solver_answer)


    # Verifier

    print("\n--- VERIFIER ---")

    verification = verify(
        question,
        solver_answer
    )

    print(f"Verdict:     {verification.verdict}")
    print(f"Confidence:  {verification.confidence:.2f}")
    print(f"Explanation: {verification.explanation}")


    # Critic

    print("\n--- CRITIC ---")

    critique = criticize(
        question,
        solver_answer,
        verification
    )

    print(critique)


    # Finalizer

    print("\n--- FINAL ANSWER ---")

    final_answer = finalize(
        question,
        solver_answer,
        verification,
        critique
    )

    print(final_answer)


    # Return same structure as Mock Mode.
    # "verifier" stays a human-readable string for API compatibility:
    # it is the formatted VerifierResult.

    return {
        "solver": solver_answer,
        "verifier": _format_verifier(verification),
        "critic": critique,
        "final": final_answer
    }