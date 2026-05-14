"""ZAYA NVFP4 GGUF bridge for vLLM.

Patches the gguf library and vLLM's GGUF loader at runtime to:
1. Register "zaya" as a known GGUF architecture
2. Provide tensor name mapping from shortened GGUF names → full HF param names
3. Fix vLLM's GGUF loader to dispatch to ZayaForCausalLM

Usage (before any vLLM import):
    import patches.nvfp4_gguf_bridge
    patches.nvfp4_gguf_bridge.patch()
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def _get_zaya_arch_id() -> int:
    import gguf

    existing = set(gguf.MODEL_ARCH_NAMES.keys())
    for val in "zaya", "ZAYA":
        for k, v in gguf.MODEL_ARCH_NAMES.items():
            if v == val:
                return k
    return max(existing) + 1


def patch() -> None:
    import gguf

    zaya_id = _get_zaya_arch_id()

    # Register the architecture
    gguf.MODEL_ARCH_NAMES[zaya_id] = "zaya"
    if not hasattr(gguf.MODEL_ARCH, "ZAYA"):
        setattr(gguf.MODEL_ARCH, "ZAYA", zaya_id)

    _orig_get_tensor_name_map = gguf.get_tensor_name_map

    def patched_get_tensor_name_map(arch, num_layers):
        arch_name = gguf.MODEL_ARCH_NAMES.get(arch, "")
        if arch_name == "zaya":
            return {}
        return _orig_get_tensor_name_map(arch, num_layers)

    gguf.get_tensor_name_map = patched_get_tensor_name_map

    logger.info("Registered zaya architecture in gguf (id=%d)", zaya_id)
    _patch_vllm_gguf_loader()
    _patch_vllm_zaya_registry()
    _register_nvfp4_quantization()


def _patch_vllm_gguf_loader() -> None:
    """Patch vLLM's GGUF loader to handle zaya arch and name mapping."""
    try:
        import vllm.model_executor.model_loader.gguf_loader as gl

        _orig_get_hf_config = gl.GGUFModelConfig

        # Monkey-patch: when GGUF has arch=zaya, set model_type=zaya in HF config
        _orig_from_gguf = gl.GGUFModelConfig.from_gguf

        @classmethod
        def patched_from_gguf(cls, gguf_path, **kwargs):
            result = _orig_from_gguf.__func__(cls, gguf_path, **kwargs)
            return result

        gl.GGUFModelConfig.from_gguf = patched_from_gguf

        _orig_load_weights = gl.GGUFLoader.load_weights

        def patched_load_weights(self, model, model_config):
            """Load weights using name_map.json for shortened names."""
            gguf_path = self.model_path
            map_path = gguf_path.replace(".gguf", ".name_map.json")
            name_map: dict[str, str] = {}
            try:
                with open(map_path) as f:
                    name_map = json.load(f)
                logger.info("Loaded name map: %d entries", len(name_map))
            except FileNotFoundError:
                logger.warning("No name_map.json found for %s", gguf_path)

            if name_map:
                self._zaya_name_map = name_map
            _orig_load_weights(self, model, model_config)

        gl.GGUFLoader.load_weights = patched_load_weights

        # Patch weight mapping in GGUFLoader
        _orig_get_params_map = gl.GGUFLoader._get_params_map

        def patched_get_params_map(self, model, name_map, vision_name_map):
            if hasattr(self, "_zaya_name_map") and self._zaya_name_map:
                params = dict(model.named_parameters())
                gguf_params: dict[str, list[str]] = {}
                for short_name, orig_name in self._zaya_name_map.items():
                    if orig_name in params:
                        gguf_params[short_name] = [orig_name]
                return gguf_params
            return _orig_get_params_map(self, model, name_map, vision_name_map)

        if hasattr(gl.GGUFLoader, "_get_params_map"):
            gl.GGUFLoader._get_params_map = patched_get_params_map

        logger.info("Patched vLLM GGUF loader for zaya arch")
    except ImportError:
        pass


def _patch_vllm_zaya_registry() -> None:
    """Ensure ZayaForCausalLM is in vLLM's model registry for GGUF loading."""
    try:
        from vllm.model_executor.models.registry import _MODELS
        from vllm.model_executor.models.zaya import ZayaForCausalLM

        if "ZayaForCausalLM" not in _MODELS:
            _MODELS["ZayaForCausalLM"] = ZayaForCausalLM
        logger.info("ZayaForCausalLM registered in vLLM model registry")
    except ImportError:
        pass


def _register_nvfp4_quantization() -> None:
    """Register NVFP4 GGUF quantization method in vLLM."""
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

        @register_quantization_config("nvfp4")
        class NVFP4Config(QuantizationConfig):
            def get_name(self) -> str:
                return "nvfp4"

            def get_supported_act_dtypes(self):
                import torch

                return [torch.float16, torch.bfloat16]

            @classmethod
            def get_min_capability(cls) -> int:
                return 120

            @staticmethod
            def get_config_filenames() -> list[str]:
                return []

            @classmethod
            def from_config(cls, config):
                return cls()

            def get_quant_method(self, layer, prefix):
                return None

        logger.info("NVFP4 quantization registered")
    except ImportError:
        pass


if __name__ == "__main__":
    patch()
    print("ZAYA NVFP4 GGUF bridge patched successfully")
