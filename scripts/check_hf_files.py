"""Check Zyphra/ZAYA1-8B files on HuggingFace."""
import json
import urllib.request

url = "https://huggingface.co/api/models/Zyphra/ZAYA1-8B"
data = json.loads(urllib.request.urlopen(url).read())
siblings = data.get("siblings", [])

print("Tokenizer-related files:")
for s in siblings:
    name = s["rfilename"]
    if "token" in name.lower() or "vocab" in name.lower() or "merge" in name.lower():
        print(f"  {name}")

print("\nAll files:")
for s in siblings:
    print(f"  {s['rfilename']}")
