#!/usr/bin/env python3
import glob
import json
import pathlib
import re
from collections import Counter

snap = glob.glob(str(
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--Ttimms--zaya1-8b-nvfp4-w4a4-uniform"
      "/snapshots/*/model.safetensors.index.json"
))[0]
ckpt = set(json.load(open(snap))["weight_map"].keys())

log = (pathlib.Path.home() / "zaya1-nvfp4-w4a4/results/cudagraph_sweep"
       "/reference_configfix.log").read_text(errors="ignore")
m = re.search(r"are: \{(.*?)\}", log, re.S)
expected = set(re.findall(r"'([^']+)'", m.group(1))) if m else set()

print("checkpoint tensors :", len(ckpt))
print("model params dumped:", len(expected))

missing = sorted(ckpt - expected)   # in checkpoint, model has no home for it
extra = sorted(expected - ckpt)     # model wants it, checkpoint lacks it

print()
print("In checkpoint but NOT a model param:", len(missing))
for n in missing[:20]:
    print("   ", n)
if len(missing) > 20:
    print("    ... +%d more" % (len(missing) - 20))

print()
print("Model param NOT in checkpoint:", len(extra))
for n in extra[:20]:
    print("   ", n)
if len(extra) > 20:
    print("    ... +%d more" % (len(extra) - 20))


def stem(n):
    return re.sub(r"\.\d+\.", ".N.", n)


print()
print("-- collapsed patterns, checkpoint-side unmatched --")
for p, c in Counter(stem(n) for n in missing).most_common(12):
    print("  %5d  %s" % (c, p))
print()
print("-- collapsed patterns, model-side unmatched --")
for p, c in Counter(stem(n) for n in extra).most_common(12):
    print("  %5d  %s" % (c, p))
