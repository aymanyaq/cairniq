"""Tests for agent/quick_actions.py — server-side quick-action prompts."""

from agent import quick_actions as qa


class _Msg:
    def __init__(self, content):
        self.content = content


def test_priority_prompt_loaded():
    assert qa.QUICK_ACTION_PROMPTS.get("priority", "").startswith("[System Instruction: Today")


def test_parse_marker_with_deep_prefix():
    assert qa.parse_quick_action_marker("[DeepReasoning] [QuickAction name=priority]") == ("priority", "")


def test_parse_marker_unknown_name_is_none():
    assert qa.parse_quick_action_marker("[QuickAction name=does_not_exist]") is None
    assert qa.parse_quick_action_marker("just a normal question") is None


def test_build_appends_focus_passthrough():
    out = qa.build_quick_action_prompt("priority", "FOCUS: ticker NVDA")
    assert out.startswith("[System Instruction:") and out.endswith("FOCUS: ticker NVDA")


def test_rewrite_marked_message_in_place():
    msgs = [_Msg("[DeepReasoning] [QuickAction name=priority]\n\nFOCUS: ticker NVDA")]
    assert qa.maybe_rewrite_quick_action(msgs) is True
    assert msgs[-1].content.startswith("[System Instruction:")
    assert "NVDA" in msgs[-1].content


def test_rewrite_noop_without_marker():
    msgs = [_Msg("Should I trim NVDA?")]
    assert qa.maybe_rewrite_quick_action(msgs) is False
    assert msgs[-1].content == "Should I trim NVDA?"


# --- required-tool floor (the prompt's mandated tools must be bindable) -------
def test_required_tools_for_priority_action():
    tools = qa.required_tools_for_action("priority")
    assert "get_market_pulse_data" in tools
    assert "check_portfolio_allocation" in tools
    assert "scan_intraday_movers" in tools
    assert "verify_portfolio_holdings" in tools
    # Registered @tool name — NOT the underlying detect_sector_rotation function name.
    assert "check_sector_rotation" in tools
    assert "detect_sector_rotation" not in tools


def test_required_tools_for_unknown_action_is_empty():
    assert qa.required_tools_for_action("does_not_exist") == []
    assert qa.required_tools_for_action("") == []


def test_priority_prompt_names_the_registered_sector_tool():
    # The prose and the floor must agree on the tool name, or the LLM is told to call a
    # tool that isn't bound under that name.
    prompt = qa.QUICK_ACTION_PROMPTS["priority"]
    assert "check_sector_rotation" in prompt
    assert "detect_sector_rotation" not in prompt
