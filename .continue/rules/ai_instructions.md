# CONTINUE IDE AGENT INSTRUCTIONS

Behavioral guidelines to reduce common LLM coding mistakes and optimize for Backend & AI development.

**Tradeoff:** These guidelines bias toward caution over speed, and surgical precision over mass refactoring.

## 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Stack-Specific Simplicity
**Minimum code that solves the problem. Adhere to Python/AI best practices.**

- **Python/Backend:** Always use strict Type Hints. Keep FastAPI route handlers or Django views lean; push business logic down. 
- **AI/ML:** When working with PyTorch or model integrations, prioritize memory-efficient code (assume VRAM constraints, e.g., <8GB). Clean up tensors and avoid unnecessary duplication.
- No features beyond what was asked. No abstractions for single-use code.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Match existing style (even if you'd do it differently).
- When resolving Git merge conflicts or integrating branches, only resolve the specific overlapping lines.
- Remove imports/variables/functions that YOUR changes made unused. Don't remove pre-existing dead code unless asked.

## 4. MCP & Tool Integration
**Leverage available tools responsibly.**

- If the workspace uses Model Context Protocol (MCP) servers, utilize them to gather system context, read database schemas, or check configurations before generating large chunks of logic.
- Do not guess project structures; use tools to list directories or read files if unsure.

## 5. Goal-Driven Execution
**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]

## STRICT CODING RULES FOR AI AGENT
1. **DO NOT INVENT VARIABLES:** Always read the provided files thoroughly. You must strictly use the existing class names, function signatures, and variable names.
2. **NO DESTRUCTIVE REWRITES:** If I ask you to add a feature or update a function, ONLY output the modified parts or clearly indicate what is appended. Do NOT rewrite the entire file unless explicitly asked.
3. **MATCH MODELS & SERIALIZERS:** Ensure that Django Serializer fields match the Django Model fields 100% perfectly to prevent `ImproperlyConfigured` errors.
4. **THINK BEFORE CODING:** Before giving code, briefly state which existing variables/functions you are going to use.