import json, re
from pathlib import Path

bad = 0
for f in Path("data/clean").glob("*.json"):
    doc = json.loads(f.read_text(encoding="utf-8"))
    for p in doc["pages"]:
        bad += len(re.findall(r"[a-z]- [a-z]", p["text"]))
print("Suspicious leftover hyphen breaks:", bad)