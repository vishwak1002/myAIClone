from datetime import datetime
import pytest

from ai_clone.ingestion.models import ParsedMessage, Platform
from ai_clone.personality.builder import PersonalityBuilder


def _msg(text: str) -> ParsedMessage:
    return ParsedMessage(
        sender="Vishwa", text=text,
        timestamp=datetime(2023, 1, 1), platform=Platform.WHATSAPP
    )


def test_empty_messages_returns_default_profile():
    builder = PersonalityBuilder()
    profile = builder.build([])
    assert profile.avg_sentence_length == 0.0
    assert profile.emoji_frequency == 0.0


def test_avg_sentence_length():
    builder = PersonalityBuilder()
    messages = [_msg("one two three"), _msg("four five")]
    profile = builder.build(messages)
    assert profile.avg_sentence_length == 2.5


def test_emoji_frequency():
    builder = PersonalityBuilder()
    messages = [_msg("hello 😀"), _msg("no emojis")]
    profile = builder.build(messages)
    assert profile.emoji_frequency == 0.5


def test_top_phrases_are_bigrams():
    builder = PersonalityBuilder()
    messages = [_msg("how are you"), _msg("how are things"), _msg("how are we")]
    profile = builder.build(messages)
    assert "how are" in profile.top_phrases


def test_informal_text_has_low_formality():
    builder = PersonalityBuilder()
    messages = [_msg("lol lmao omg brb tbh ngl idk imo gonna wanna")]
    profile = builder.build(messages)
    assert profile.formality_score < 0.5


def test_formal_text_has_high_formality():
    builder = PersonalityBuilder()
    messages = [_msg("I would like to discuss the proposal thoroughly.")]
    profile = builder.build(messages)
    assert profile.formality_score > 0.5


def test_punctuation_style_sums_to_one_or_zero():
    builder = PersonalityBuilder()
    messages = [_msg("Hello! How are you? I'm fine.")]
    profile = builder.build(messages)
    total = sum(profile.punctuation_style.values())
    assert abs(total - 1.0) < 1e-6 or total == 0.0
