from dotenv import load_dotenv
from google import genai
import time


# ============================================================
# GEMINI SETUP
# ============================================================

load_dotenv()

client = genai.Client()


# ============================================================
# GEMINI REQUEST HANDLER
# ============================================================

def ask_gemini(prompt):

    for attempt in range(3):

        try:

            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt
            )

            return response.output_text

        except Exception as e:

            error_text = str(e)

            if (
                "503" in error_text
                or "service_unavailable" in error_text
            ):

                print(
                    f"\nGemini temporarily unavailable. "
                    f"Retrying... ({attempt + 1}/3)"
                )

                time.sleep(5)

            else:
                raise

    raise RuntimeError(
        "Gemini is still unavailable after 3 attempts."
    )


# ============================================================
# SOLVER
# ============================================================

def solve(question):

    return ask_gemini(f"""
You are the Solver in an AI verification system.

Solve the user's question carefully and logically.

Show the necessary reasoning and calculations.

Do not intentionally make mistakes.

User question:
{question}
""")


# ============================================================
# VERIFIER
# ============================================================

def verify(question, solver_answer):

    return ask_gemini(f"""
You are the Verifier in an AI verification system.

Independently check the Solver's answer.

Check:

1. Is the answer correct?
2. Is the reasoning correct?
3. Are there calculation errors?
4. Are important details or assumptions missing?
5. Are the equations and units correct?

Do NOT blindly trust the Solver.

User question:
{question}

Solver's answer:
{solver_answer}

Give your verdict as PASS or FAIL, followed by a concise explanation.
""")


# ============================================================
# CRITIC
# ============================================================

def criticize(question, solver_answer, verification):

    return ask_gemini(f"""
You are the Critic in an AI verification system.

Your job is to critically examine the Solver's answer
and the Verifier's assessment.

Look for:

1. Incorrect reasoning
2. Calculation errors
3. Incorrect equations
4. Unit errors
5. Missing information
6. Hidden assumptions
7. Ambiguity
8. Problems the Verifier may have missed

Do not criticize something merely because you would phrase
it differently.

User question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification}

Give a concise critique.

If there is no meaningful problem, say:

NO SIGNIFICANT ISSUES
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

    return ask_gemini(f"""
You are the Finalizer in an AI verification system.

Produce the final answer for the user.

You have:

- Original question
- Solver's answer
- Verifier's assessment
- Critic's analysis

Rules:

1. Do not blindly trust the Solver.
2. Correct the Solver if necessary.
3. Consider the Verifier's assessment.
4. Consider the Critic's concerns.
5. Give a clear and accurate final answer.
6. Include appropriate calculations and units when needed.
7. Do not mention the internal AI agents unless necessary.

Original question:
{question}

Solver's answer:
{solver_answer}

Verifier's assessment:
{verification}

Critic's analysis:
{critique}

Now produce the final answer.
""")


# ============================================================
# GEMINI PIPELINE
# ============================================================

def run_gemini_pipeline(question):

    # Solver
    solver_answer = solve(question)

    print("\n--- SOLVER ---")
    print(solver_answer)


    # Verifier
    verification = verify(
        question,
        solver_answer
    )

    print("\n--- VERIFIER ---")
    print(verification)


    # Critic
    critique = criticize(
        question,
        solver_answer,
        verification
    )

    print("\n--- CRITIC ---")
    print(critique)


    # Finalizer
    final_answer = finalize(
        question,
        solver_answer,
        verification,
        critique
    )

    print("\n--- FINAL ANSWER ---")
    print(final_answer)


    # Return same structure as Mock Mode
    return {
        "solver": solver_answer,
        "verifier": verification,
        "critic": critique,
        "final": final_answer
    }