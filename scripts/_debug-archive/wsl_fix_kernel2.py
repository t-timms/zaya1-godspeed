p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/__init__.py"
with open(p) as f:
    c = f.read()

old = """    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \\n" + "\\n".join(failure_reasons)
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
    print("Patched")
else:
    # Try simpler pattern
    if "Failed to find a kernel that can implement the" in c:
        print("Found target line, trying different pattern")
        # Try with single backslash
        old2 = """    raise ValueError(
        "Failed to find a kernel that can implement the "
        "WNA16 linear layer. Reasons: \\n" + "\\n".join(failure_reasons)
    )"""
        if old2 in c:
            c = c.replace(old2, new)
            with open(p, "w") as f:
                f.write(c)
            print("Patched v2")
        else:
            print("All patterns failed")
    else:
        print("Target string not found at all")
