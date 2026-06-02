"""Download ZAYA1-8B tokenizer files from HuggingFace."""

import os

from huggingface_hub import hf_hub_download

model_dir = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
os.makedirs(model_dir, exist_ok=True)

files_to_download = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

print("Downloading tokenizer files from Zyphra/ZAYA1-8B...")
for filename in files_to_download:
    try:
        local_path = hf_hub_download(
            repo_id="Zyphra/ZAYA1-8B",
            filename=filename,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  Downloaded: {filename}")
    except Exception as e:
        print(f"  Failed: {filename} - {e}")

print("\nFiles in model directory:")
for f in sorted(os.listdir(model_dir)):
    path = os.path.join(model_dir, f)
    size = os.path.getsize(path)
    print(f"  {f} ({size} bytes)")
print("Done.")
