"""[cachePoint] markers must mean something on every provider that has caching.

The markers were honoured only when `_is_bedrock_provider()`; every other
provider flattened them away, so ~7.7k tokens of stable prompt prefix were
re-billed on every call.

Bedrock and Anthropic express the same idea inversely — Bedrock inserts a
separator block and caches everything BEFORE it; Anthropic tags the LAST block of
the cached prefix with `cache_control`. Google/Vertex has no marker at all: it
caches implicitly from a stable prefix, so the correct handling there is still to
strip, and a literal marker would corrupt the very prefix being matched.
"""
import pytest

from agent.utils import _anthropic_cache_blocks, _system_prompt_message

STRUCTURED = [
    {"text": "STATIC RULES: a long stable preamble."},
    {"cachePoint": {"type": "default"}},
    {"text": "VOLATILE: today's portfolio."},
]


def test_anthropic_marks_the_end_of_the_cached_prefix():
    blocks = _anthropic_cache_blocks(STRUCTURED)
    assert [b["text"] for b in blocks] == [
        "STATIC RULES: a long stable preamble.",
        "VOLATILE: today's portfolio.",
    ]
    # The marker attaches to the block BEFORE it, and does not become a block.
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]


def test_no_cache_control_leaks_onto_the_volatile_tail():
    """Caching the tail would invalidate the entry on every turn."""
    blocks = _anthropic_cache_blocks(STRUCTURED)
    assert "cache_control" not in blocks[-1]


def test_anthropic_breakpoints_are_capped_at_four():
    many = []
    for i in range(8):
        many.append({"text": f"section {i}"})
        many.append({"cachePoint": {"type": "default"}})
    blocks = _anthropic_cache_blocks(many)
    marked = [b for b in blocks if "cache_control" in b]
    # More than four is a hard API error, so the extras must be dropped, not sent.
    assert len(marked) == 4
    # The earliest breakpoints are the ones kept — they cover the largest prefix.
    assert [b["text"] for b in marked] == [f"section {i}" for i in range(4)]


def test_a_leading_marker_with_nothing_before_it_is_ignored():
    blocks = _anthropic_cache_blocks([{"cachePoint": {"type": "default"}}, {"text": "body"}])
    assert blocks == [{"type": "text", "text": "body"}]


@pytest.mark.parametrize("provider", ["google", "vertexai", "openai", "azure"])
def test_providers_without_explicit_markers_get_flat_text(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    msg = _system_prompt_message(STRUCTURED)
    assert isinstance(msg.content, str)
    # The marker must not survive as literal text: on Gemini that would change
    # the prefix bytes and defeat the implicit cache it is meant to help.
    assert "cachePoint" not in msg.content
    assert "STATIC RULES" in msg.content and "VOLATILE" in msg.content


def test_bedrock_keeps_the_structured_blocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    msg = _system_prompt_message(STRUCTURED)
    assert msg.content == STRUCTURED


def test_anthropic_gets_blocks_not_a_flat_string(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    msg = _system_prompt_message(STRUCTURED)
    assert isinstance(msg.content, list)
    assert msg.content[0]["cache_control"] == {"type": "ephemeral"}


def test_a_plain_string_prompt_is_untouched_on_every_provider(monkeypatch):
    for provider in ("bedrock", "anthropic", "vertexai"):
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert _system_prompt_message("just text").content == "just text"
