#!/usr/bin/env python3
"""Do the published ignore regexes still mean the same thing on the REFACTORED base?

The regexes were written against the legacy 80-layer module names. The refactored
base renames modules, so a pattern can silently start matching nothing (leaving a
sensitive module quantized) or start matching something new (over-excluding).
This is pure name analysis against the safetensors index. No GPU, no model load.
"""
import glob
import json
import pathlib
import re
from collections import Counter

IGNORE = [
    "lm_head",
    r"re:.*router.*",
    r"re:.*norm.*",
    r"re:.*qkv.*",
    r"re:.*cca.*",
]


def load_names(pattern):
    hits = glob.glob(pattern)
    if not hits:
        return None
    return set(json.load(open(hits[0]))["weight_map"].keys())


HUB = pathlib.Path.home() / ".cache/huggingface/hub"
new = load_names(str(HUB / "models--Zyphra--ZAYA1-8B/snapshots/*/model.safetensors.index.json"))
old = load_names(str(HUB / "models--Ttimms--zaya1-8b-nvfp4-w4a4-uniform/snapshots/*"
                          "/model.safetensors.index.json"))

if new is None:
    raise SystemExit("refactored base index not cached yet")

# A Linear is approximated by a module owning a `.weight`; drop the quantization
# side-tensors so we compare module inventories, not artifacts of the old export.
SIDE = ("_scale", "_packed", "_global_scale", "weight_scale", "input_global_scale")


def modules(names):
    out = set()
    for n in names:
        if any(n.endswith(s) for s in SIDE):
            continue
        if n.endswith(".weight"):
            out.add(n[: -len(".weight")])
    return out


new_mods = modules(new)


def matches(mod, pat):
    if pat.startswith("re:"):
        return re.search(pat[3:], mod) is not None
    return pat in mod


print(f"refactored base: {len(new)} tensors, {len(new_mods)} weight-owning modules")
print()
print("Pattern coverage on the REFACTORED base")
print("-" * 62)
dead = []
for pat in IGNORE:
    hit = [m for m in new_mods if matches(m, pat)]
    flag = "  <-- MATCHES NOTHING" if not hit else ""
    print(f"  {pat:20s} -> {len(hit):5d} modules{flag}")
    if not hit:
        dead.append(pat)

ignored = {m for m in new_mods if any(matches(m, p) for p in IGNORE)}
targeted = new_mods - ignored
print()
print(f"  would be IGNORED : {len(ignored)}")
print(f"  would be QUANTED : {len(targeted)}")

print()
print("Sample of what WOULD be quantized (collapsed):")
stem = lambda n: re.sub(r"\.\d+\.", ".N.", n)  # noqa: E731
for p, c in Counter(stem(m) for m in targeted).most_common(14):
    print(f"  {c:5d}  {p}")

print()
print("CCA / conv modules in the refactored base, and whether they are excluded:")
conv = sorted({stem(m) for m in new_mods if "conv" in m or "cca" in m or "delayed" in m})
for m in conv:
    example = next(x for x in new_mods if stem(x) == m)
    who = [p for p in IGNORE if matches(example, p)]
    verdict = f"excluded by {who}" if who else "*** QUANTIZED ***"
    print(f"  {m:58s} {verdict}")

if dead:
    print()
    print("DEAD PATTERNS (matched nothing on the refactored base):")
    for p in dead:
        print(f"  {p}")

if old:
    print()
    old_mods = modules(old)
    print(f"legacy checkpoint modules: {len(old_mods)}  |  refactored: {len(new_mods)}")
