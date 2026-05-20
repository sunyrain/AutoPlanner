"""Small stable identifiers used by lightweight cascade-search modules."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

