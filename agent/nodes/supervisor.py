import os
import re

from langchain_core.messages import HumanMessage

from agent.constants import HEALTH_CHECK_KEYWORDS
from agent.dspy_setup import DSPY_AVAILABLE, configure_dspy

# --- Logging ---
from agent.logger import log_event
from agent.state import AgentState
from agent.utils import safe_print, send_status

# Conservative markers that identify a user-pasted system-instruction / framework
# block. Any one of these is enough to capture the message as a sticky framework
# that gets replayed in every subsequent synthesis prompt. Keep the markers
# narrow on purpose — false positives would freeze the agent into a template
# the user did not intend to be persistent.
_FRAMEWORK_HEADER_MARKERS = (
    "[system instruction",
    "[system prompt",
    "[instruction:",
    "[instructions:",
    "<system>",
    "<instructions>",
    "<system_instruction",
)

_FRAMEWORK_INLINE_MARKERS = (
    ("role & context:", "required output format"),
    ("role & context:", "output format"),
    ("stage 1", "required output format"),
    ("stage 1:", "stage 2"),
    ("<role>", "<output_format>"),
    ("you are an", "required output format"),
    ("you are a ", "required output format"),
)


def _extract_user_framework(content: str) -> str:
    """Detect whether the user message embeds a system-instruction block.

    Returns the framework text (the whole user message, since the framework
    usually IS the whole message) when matched, otherwise an empty string.

    Conservative: only triggers on clear header markers or on combinations of
    structural keywords that almost never appear in a natural question.
    """
    if not content:
        return ""
    text = str(content).strip()
    if len(text) < 200:
        # Real framework blocks are always substantial. Anything shorter is
        # almost certainly a question or chat, not an instruction template.
        return ""

    lower_head = text[:400].lower()
    for marker in _FRAMEWORK_HEADER_MARKERS:
        if marker in lower_head:
            return text

    lower_all = text.lower()
    for marker_a, marker_b in _FRAMEWORK_INLINE_MARKERS:
        if marker_a in lower_all and marker_b in lower_all:
            return text

    return ""

# --- LLM Config ---
# Module import must NOT crash when AIDLC_MODEL_ID is missing — for Anthropic/OpenAI
# providers, model resolution happens lazily inside agent.utils.get_llm().
MODEL_ID = os.environ.get("AIDLC_MODEL_ID")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# DSPy is provider-agnostic: configure_dspy() builds the LiteLLM-backed LM for
# whichever LLM_PROVIDER is active (bedrock/openai/anthropic/google/azure).
if DSPY_AVAILABLE:
    configure_dspy(MODEL_ID, REGION, error_callback=safe_print)

def supervisor_node(state: AgentState):
    """
    Supervisor: Routes based on conversation history.
    As requested: Defaults all manual chat to DeepReasoning, while
    honoring explicit @mentions for specialist buttons.
    """
    messages = state['messages']
    last_msg = messages[-1] if messages else None

    # --- STICKY USER FRAMEWORK ---
    # If the latest user message contains a system-instruction / framework
    # block, capture it so downstream synthesis nodes can replay its output
    # format on every subsequent turn. Once captured, it persists in checkpoint
    # state until a new framework block replaces it. Same-message overrides
    # win (the user can paste a different framework to switch templates).
    existing_framework = state.get("user_framework", "") or ""
    new_user_framework = existing_framework
    if last_msg and isinstance(last_msg, HumanMessage):
        last_content = last_msg.content
        if isinstance(last_content, list):
            last_content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in last_content
            )
        detected = _extract_user_framework(str(last_content or ""))
        if detected:
            if detected != existing_framework:
                new_user_framework = detected
                log_event("Supervisor", "Captured user framework block", {
                    "framework_chars": len(detected),
                    "had_previous_framework": bool(existing_framework),
                })
                send_status("📐 Captured custom analysis framework — will replay on Deep Reasoning turns.")

    # 0. EXPLICIT OVERRIDES / SUMMONING (for buttons)
    if last_msg and isinstance(last_msg, HumanMessage):
        content = str(last_msg.content)
        if "@DeepReasoning" in content or "[DeepReasoning]" in content:
            send_status("🧠 Routing to Deep Reasoning Engine...")
            log_event("Supervisor", "Explicit Override: DeepReasoning")
            return {"next": "DeepReasoning", "summary": state.get("summary", ""), "user_framework": new_user_framework}
        if (
            "@MarketAnalyst" in content
            or "[MarketAnalyst]" in content
            or "[MarketAnalyst " in content  # supports quick-action lenses: [MarketAnalyst lens=...]
        ):
            send_status("📊 Routing to Market Analyst...")
            log_event("Supervisor", "Explicit Override: MarketAnalyst")
            return {"next": "MarketAnalyst", "summary": state.get("summary", ""), "user_framework": new_user_framework}
        if "@PortfolioManager" in content or "[PortfolioManager]" in content:
            send_status("💼 Routing to Portfolio Manager...")
            log_event("Supervisor", "Explicit Override: PortfolioManager")
            return {"next": "PortfolioManager", "summary": state.get("summary", ""), "user_framework": new_user_framework}
        if "@NewsAnalyst" in content or "[NewsAnalyst]" in content:
            send_status("📰 Routing to News Analyst...")
            log_event("Supervisor", "Explicit Override: NewsAnalyst")
            return {"next": "NewsAnalyst", "summary": state.get("summary", ""), "user_framework": new_user_framework}

    # 1. FIXED FLOWS & SAFETY
    last_human_idx = -1
    for i, msg in enumerate(messages):
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
        if msg_type == "human":
            last_human_idx = i

    if last_human_idx == -1:
        log_event("Supervisor", "FINISH (No human message found)")
        return {"next": "FINISH", "summary": state.get("summary", ""), "user_framework": new_user_framework}

    messages_since_human = messages[last_human_idx + 1:]

    # MEMORY HOOK: Process user input ONLY on the first pass
    new_summary = state.get("summary", "")
    if len(messages_since_human) == 0:
        # Approximate context size by character count
        total_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
        # Summarize if we have 5+ messages (multi-turn drift) OR context is massive (>8000 chars)
        # But we also want to wait until at least the first follow-up (len >= 3)
        if len(messages) >= 3 and (len(messages) >= 5 or total_chars > 8000):
            from agent.memory import update_context_summary
            send_status("📝 Updating conversation context summary...")
            new_summary, summary_ok = update_context_summary(messages, state.get("summary", ""))
            if summary_ok:
                log_event("Supervisor", "Context Summary Updated", {"messages": len(messages), "chars": total_chars})
            else:
                log_event("Supervisor", "Context Summary Update Failed (kept previous)", {"messages": len(messages), "chars": total_chars})

        human_msg = messages[last_human_idx]
        content = human_msg.get("content") if isinstance(human_msg, dict) else getattr(human_msg, "content", "")
        if content:
            # Check for Privacy Mode Signal (State flag OR text tags). The tag
            # list is shared with the chat router's feedback capture — see
            # agent.utils.PRIVACY_TAGS.
            from agent.utils import is_private_turn

            state_ghost = state.get("ghost", False)
            is_private = is_private_turn(content, state_ghost)

            if is_private:
                send_status("🛡️ Ghost Mode Active: Conversation will not be recorded in memory.")
                log_event("Supervisor", "Privacy Mode Enabled: Skipping memory capture", {"source": "state" if state_ghost else "text"})
            else:
                try:
                    from tools.memory import process_user_message
                    process_user_message(str(content))
                except Exception as e:
                    safe_print(f"⚠️ Memory processing failed: {e}")

    # Safety Valve: Prevent infinite loops
    if len(messages_since_human) > 10:
        send_status("⚠️ Safety Valve Triggered: Turn limit reached.", degraded=True)
        log_event("Supervisor", "FINISH (Safety Valve Triggered)", {"turns": len(messages_since_human)})
        return {"next": "FINISH", "summary": new_summary, "user_framework": new_user_framework}


    worker_names = {"PortfolioManager", "NewsAnalyst", "MarketAnalyst", "RiskManager", "DeepReasoning"}
    responded_this_turn = set()

    for msg in messages[last_human_idx + 1:]:
        def safe_get(obj, key):
            if isinstance(obj, dict): return obj.get(key)
            val = getattr(obj, key, None)
            if key == 'name' and not val:
                kwargs = getattr(obj, 'additional_kwargs', {})
                val = kwargs.get('name')
            return val

        name = safe_get(msg, 'name')
        content = safe_get(msg, 'content')
        if isinstance(content, list):
             # Handle Bedrock list content
             content = "\n".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])

        if name and name in worker_names:
            responded_this_turn.add(name)

        # Check content prefix e.g. [DeepReasoning]
        if content:
            content_str = str(content).strip()
            for w_name in worker_names:
                if content_str.startswith(f"[{w_name}]") or f"[{w_name}]:" in content_str[:40]:
                    responded_this_turn.add(w_name)


    # --- RISK MANAGEMENT & TERMINATION ---
    if "RiskManager" in responded_this_turn:
         log_event("Supervisor", "FINISH (Risk analysis complete)")
         return {"next": "FINISH", "summary": new_summary, "user_framework": new_user_framework}

    # --- HEALTH CHECK BYPASS: Skip RiskManager for health check queries ---
    # Scoped to the current turn (the triggering human message onward) so a
    # health-check phrase from an earlier turn can't keep bypassing the
    # compliance gate on later, unrelated advice queries.
    is_health_check = False
    for msg in messages[last_human_idx:]:
        msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
        if msg_type == "tool":
            # Raw tool output is untrusted data, not user intent — e.g.
            # scan_opportunities always returns a dict with a literal
            # "diagnostics" key, which would otherwise false-positive this
            # check on every scan turn and skip RiskManager unconditionally.
            continue
        content = str(getattr(msg, "content", msg)).lower()
        if any(keyword in content for keyword in HEALTH_CHECK_KEYWORDS):
            is_health_check = True
            break

    if is_health_check and any(w in responded_this_turn for w in ["DeepReasoning", "MarketAnalyst", "PortfolioManager"]):
        log_event("Supervisor", "FINISH (Health check complete - skipping RiskManager)")
        return {"next": "FINISH", "summary": new_summary, "user_framework": new_user_framework}

    if any(w in responded_this_turn for w in ["DeepReasoning", "MarketAnalyst", "PortfolioManager"]):
         send_status("⚖️ Routing to Risk Manager for compliance review...")
         log_event("Supervisor", "Routing to RiskManager (Pre-finish check)")
         return {"next": "RiskManager", "summary": new_summary, "user_framework": new_user_framework}

    if "NewsAnalyst" in responded_this_turn:
         log_event("Supervisor", "FINISH (Market context provided)")
         return {"next": "FINISH", "summary": new_summary, "user_framework": new_user_framework}

    # --- DEFAULT ROUTING ---
    send_status("🧠 Routing to DeepReasoning")
    log_event("Supervisor", "Routing to DeepReasoning (Initial Triage)")
    return {"next": "DeepReasoning", "summary": new_summary, "user_framework": new_user_framework}
