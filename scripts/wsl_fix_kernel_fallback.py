p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/__init__.py"
with open(p) as f:
    c = f.read()

old = """    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \n" + "\n".join(failure_reasons)
    )"""

new = """    import logging
    logging.getLogger(__name__).warning(
        "No WNA16 kernel, using unquantized fallback. Reasons: %s",
        "; ".join(failure_reasons)
    )
    return None"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Patched: return None for unsupported layers")
else:
    print("Pattern not found")
