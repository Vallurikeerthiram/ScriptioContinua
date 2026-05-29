# Local Model Notes

Laptop profile detected while preparing this pipeline:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8 GB
- RAM: about 16 GB total
- Disk free: about 381 GB
- Ollama: 0.21.1

Recommended local Ollama tags for this machine:

- `qwen3.5:4b`
- `gemma4:e2b`

Requested models with blockers:

- `deepseek-v3.2` is available in Ollama as `deepseek-v3.2:cloud`, which is cloud-only rather than a normal local download.
- `llama4` starts at `llama4:scout`, which is much larger than this laptop can run comfortably as a local Ollama model.

The scripts in `type 1`, `type 2`, and `type 3` are model-agnostic, so if you later use a remote DeepSeek setup or a stronger workstation, you can switch the `--model` value without changing the dataset pipeline.
