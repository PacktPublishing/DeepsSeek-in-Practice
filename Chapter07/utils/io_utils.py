import json
from pathlib import Path
from typing import Any, Dict, List

from .custom_exceptions import DatasetError


def write_jsonl_file(data: List[Dict[str, Any]], filepath: Path) -> None:
    """Write a list of dictionaries to a JSONL file."""
    try:
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    except Exception as e:
        raise DatasetError(
            f"Failed to write JSONL file: {str(e)}",
            file_path=str(filepath),
            expected_format="JSONL",
        ) from e
