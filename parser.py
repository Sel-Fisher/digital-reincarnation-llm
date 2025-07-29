import json
import os
from typing import Dict, List


def save_speaker_messages(dialog_data: Dict[str, List[Dict[str, str]]], speaker: str) -> str:
    """
    Filters messages from a dialog for a specific speaker and saves them to a JSON file.
    Args:
        dialog_data: dict with dialog data.
        speaker: speaker name we save.
    Returns:
        Output file path.
    """
    output_dir = "./data"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{speaker}_messages.json")

    speaker_messages = [
        msg for msg in dialog_data.get("dialog", []) if msg.get("speaker") == speaker
    ]

    if not speaker_messages:
        raise ValueError(f"No messages found for speaker: {speaker}")

    output_data = {"messages": speaker_messages}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_path