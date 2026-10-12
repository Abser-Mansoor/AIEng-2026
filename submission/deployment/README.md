# MiniGPT - a Transformer language model built from scratch

A small GPT-style, character-level causal language model (attention, multi-head attention, sinusoidal
positions, residual blocks, LayerNorm, FFN, LM head) implemented in PyTorch **without any pretrained model**,
trained on Tiny Shakespeare.

## Repository layout
| Path | What it is |
|---|---|
| `AI_Engineering_Transformer_GPT.ipynb` | Full notebook: attention, multi-head, positions, blocks, training, ablations, packaging |
| `deploy/minigpt.py` | Standalone model + tokenizer + checkpoint loader (inference only) |
| `deploy/generate.py` | Command-line interface |
| `deploy/api.py` | FastAPI service |
| `deploy/minigpt_best.pt` | Trained checkpoint (TODO: ~13 MB; or host it and set `CHECKPOINT_URL`) |
| `outputs/figures`, `outputs/logs`, `outputs/samples` | Figures, metrics/ablation CSVs, generated samples |

## Dataset
Tiny Shakespeare (~1.1 MB, ~40k lines), the corpus from Karpathy's `char-rnn`
(https://github.com/karpathy/char-rnn), also on Hugging Face as "tiny_shakespeare". Downloaded automatically by the notebook.
Character-level tokenizer, 65-character vocabulary, no special tokens. 90/10 contiguous train/validation split.

## Setup
```bash
pip install -r deploy/requirements.txt
```

## Training (reproduce the checkpoint)
Open the notebook in Google Colab (GPU runtime) and run Milestones 1-5 top to bottom.
Config used: d_model=256, n_heads=4, n_layers=4, ffn_dim=1024, context=128, dropout=0.1, AdamW, lr 6e-4 (cosine, 200 warm-up), batch 64, 3000 steps.
TODO: paste final parameter count, hardware, wall-clock time, train/val loss.

## Inference - CLI
```bash
python deploy/generate.py --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8
```
Expected output: the prompt, a Shakespeare-style continuation, then parameter count, context length, and temperature.
Use `--temperature 0` for greedy decoding, `--top-k 40` for top-k sampling, `--seed` for reproducibility.

## Inference - REST API
```bash
cd deploy
uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"prompt": "ROMEO:", "max_new_tokens": 100, "temperature": 0.8, "seed": 1}'
```
Interactive docs at `/docs`. Health check at `/health`.

## Deployment on Contabo
1. Push the repository (including `deploy/minigpt_best.pt`, or set `CHECKPOINT_URL`) to GitHub.
2. Pull the repository on the Contabo server.
3. Build the Docker image and start the server using `docker-compose.yml`

## Tests
The notebook runs every `test_*` function (attention shapes, causal masking, target shifting, end-to-end causality,
checkpoint reload, exported-module equivalence) and a tiny-overfit gate.

## Known limitations
- Character-level, 65-character vocabulary: digits and most symbols are silently dropped from prompts.
- Trained on ~1 MB of one author; output imitates the style but is not factual or semantically coherent over long spans.
- Greedy decoding loops; high temperature invents words (see the failure analysis in the report).
- Context window is 128 characters; earlier text is forgotten during longer generations.
