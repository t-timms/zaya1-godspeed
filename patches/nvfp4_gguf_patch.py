"""Monkey-patch vLLM GGUF loader to support ZAYA architecture.

Adds 'zaya' to gguf.MODEL_ARCH_NAMES and provides identity tensor name
mapping so that GGUF files with arch='zaya' and HF-style tensor names
load into ZayaForCausalLM.
"""

from __future__ import annotations

import json
from typing import Any


def patch() -> None:
    import gguf

    # Add "zaya" to the gguf architecture registry
    zaya_arch = max(gguf.MODEL_ARCH_NAMES.keys()) + 1
    gguf.MODEL_ARCH_NAMES[zaya_arch] = "zaya"
    gguf.MODEL_ARCH.ZAYA = zaya_arch

    # Store original get_tensor_name_map
    _orig_get_tensor_name_map = gguf.get_tensor_name_map

    def patched_get_tensor_name_map(arch: int, num_layers: int) -> dict[str, dict[str, Any]]:
        if gguf.MODEL_ARCH_NAMES.get(arch) == "zaya":
            # Return identity mapping — GGUF tensor names ARE HF param names
            # (they were stored with shortened names; use the name_map for reverse)
            return {}
        return _orig_get_tensor_name_map(arch, num_layers)

    gguf.get_tensor_name_map = patched_get_tensor_name_map

    # Patch vLLM's GGUF loader to handle zaya architecture
    try:
        import vllm.model_executor.model_loader.gguf_loader as gl
        _orig_init = gl.GGUFLoader.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _orig_init(self, *args, **kwargs)
            # Load name map from GGUF if available
            gguf_path = kwargs.get("model_path", args[0] if args else "")
            if hasattr(self, "_name_map"):
                return
            self._name_map: dict[str, str] = {}
            map_path = gguf_path.replace(".gguf", ".name_map.json")
            try:
                with open(map_path) as f:
                    self._name_map = json.load(f)
            except FileNotFoundError:
                pass

        gl.GGUFLoader.__init__ = patched_init
    except ImportError:
        pass


def get_model_class_for_zaya():
    """Return the ZayaForCausalLM class for GGUF loading."""
    try:
        from vllm.model_executor.models.zaya import ZayaForCausalLM
        return ZayaForCausalLM
    except ImportError:
        return None
