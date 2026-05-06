from datetime import datetime
from ai_clone.ingestion.models import ParsedMessage, Platform


def test_parsed_message_stores_fields():
    msg = ParsedMessage(
        sender="Vishwa",
        text="Hello there",
        timestamp=datetime(2023, 5, 23, 14, 32, 10),
        platform=Platform.WHATSAPP,
    )
    assert msg.sender == "Vishwa"
    assert msg.text == "Hello there"
    assert msg.platform == Platform.WHATSAPP
    assert msg.timestamp == datetime(2023, 5, 23, 14, 32, 10)


def test_platform_values():
    assert Platform.WHATSAPP == "whatsapp"
    assert Platform.TELEGRAM == "telegram"
    assert Platform.IMESSAGE == "imessage"
    assert Platform.SLACK == "slack"
