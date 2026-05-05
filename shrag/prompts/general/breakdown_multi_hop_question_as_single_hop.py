DEFAULT_PROMPT = {
    "decompose": (
        """
You are an expert question decomposer.
Given a potentially multi-hop question (and optional context), produce the **minimal** set of independent, single-hop questions that, when answered, enable a solver to answer the original question.

Requirements:
- Each item must be answerable without external tools other than the given context and common knowledge.
- Avoid redundancy; keep the set as small as possible while sufficient.
- Prefer entity- and relation-focused questions.
- Do NOT answer anything. Only output the questions.
- Output strict JSON only, no extra text.

Return JSON with this schema:
{
  "single_hop_questions": ["q1", "q2", ...]
}
"""
    ).strip(),
    "chain": (
        """
You are an expert question planner.
Decompose the given multi-hop question into an **ordered** chain of single-hop questions such that each step leads naturally to the next and ultimately to the final answer.

Rules:
- Each question should be solvable on its own given prior steps and provided context.
- Keep the chain concise (3-6 steps typical). No answers, only questions.
- Output strict JSON only.

Schema:
{
  "steps": [
    {"index": 1, "question": "..."},
    {"index": 2, "question": "..."}
  ]
}
"""
    ).strip(),
    "rewrite": (
        """
You are an expert question rewriter.
Rewrite the given multi-hop question into a single, simpler **single-hop** question that preserves the original target and remains answerable (ideally with the provided context).

Rules:
- Keep core semantics and target unchanged.
- Remove multi-hop dependencies by substituting intermediate facts directly if they are present in the context. If not, reformulate to a single relation query.
- Output strict JSON only.

Schema:
{
  "single_hop_question": "..."
}
"""
    ).strip(),
}
