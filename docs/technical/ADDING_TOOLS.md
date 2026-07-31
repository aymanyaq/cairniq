# Adding Tools

## 1. Add the capability

Create or extend a function in `tools/` that returns structured, predictable data.

Guidelines:

- Return dictionaries or short strings.
- Prefer explicit `"error"` keys instead of raising for expected data failures.
- Keep network access and parsing inside the tool layer.

## 2. Expose it to the agent

Wrap the capability in `agent/tool_registry.py` with `@tool`.

Guidelines:

- Give the tool a clear name that matches how prompts will reference it.
- Keep the description action-oriented so the model can choose it accurately.
- Add the tool to the appropriate exported tool list.

## 3. Update prompt contracts

If a node prompt mentions the new tool by name, use the exact registry name.

Good:

- `scan_opportunities`
- `get_insider_activity`

Avoid:

- Referring to the implementation function if the exposed tool name is different.

## 4. Add a regression test

At minimum, add one test that validates:

- the tool can be imported
- the tool returns the expected shape for a mocked or deterministic path
- the node or retriever path that depends on it can still run
