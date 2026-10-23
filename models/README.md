# Base model provenance

## Qwen/Qwen2.5-0.5B-Instruct

- model_id: `Qwen/Qwen2.5-0.5B-Instruct`
- revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- license: Apache-2.0
- weights: `models/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775/`
  (tracked with git-lfs)
- model.safetensors: 988 097 824 bytes,
  sha256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
  (verified against the Hub digest by the reference repo's runtime probe
  `experiments/runtime_probe_20261022/`, commit `1cd21a3`)

Runtime verified on this machine class (Apple M4 / 16 GiB / MPS, BF16, ~64 tok/s
generation) in the reference repo. Local-only loading, `trust_remote_code=False`,
safetensors-only.

## Fetch (if LFS pull is undesirable)

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --local-dir models/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775
sha256sum models/Qwen2.5-0.5B-Instruct/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors
# must equal fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
```
