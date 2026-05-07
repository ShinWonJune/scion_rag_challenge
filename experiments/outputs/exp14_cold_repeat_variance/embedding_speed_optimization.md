# Embedding Speed Optimization Recommendations

Context: Exp12/Exp13 use `shrag/features/embedding_processor.py`, which currently loads `SentenceTransformer(model_name, trust_remote_code=True, truncate_dim=..., device=...)` and calls `model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)` with no explicit batch size, model dtype, maximum token length, attention backend, or length-aware batching. The observed cold timings are therefore measuring default SentenceTransformers/PyTorch behavior plus model download/cache/GPU warmup variance.

## Current bottleneck and defaults

- `SentenceTransformer.encode()` defaults `batch_size=32` and `precision="float32"`; SentenceTransformers documents `batch_size` as an important speed knob whose optimum depends on hardware, model size, precision, and input length, so the current code is not tuned for this corpus or GPU. Source: <https://www.sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.SentenceTransformer.encode>
- `SentenceTransformer(...)` supports `model_kwargs`, `tokenizer_kwargs`, `backend={"torch","onnx","openvino"}`, and `truncate_dim`; SHRAG currently only uses `truncate_dim`, so dtype/backend/tokenizer settings are left implicit. Source: <https://www.sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.SentenceTransformer>
- The current Exp12/Exp13 GTE config uses `Alibaba-NLP/gte-multilingual-base` at 768 dimensions; BGE-M3 configs use `BAAI/bge-m3` at 1024 dimensions. BGE-M3 is substantially larger/longer-context than GTE, so it should not be expected to match GTE cold speed without aggressive max-length and precision controls.

## Recommended SHRAG config knobs

Add optional encoder config fields and thread them into `generate_batch_embeddings()` / `generate_batch_embeddings_cached()` instead of hard-coding defaults:

```jsonc
{
  "model_name": "Alibaba-NLP/gte-multilingual-base",
  "embedding_dim": 768,
  "embedding_mode": "3T+A",
  "batch_size": 64,
  "torch_dtype": "float16",
  "max_seq_length": 512,
  "normalize_embeddings": true,
  "encode_precision": "float32",
  "backend": "torch",
  "attn_implementation": "sdpa",
  "length_bucket": true
}
```

Suggested implementation mapping:

```python
model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    truncate_dim=truncate_dimension,
    device=device,
    backend=backend,
    model_kwargs={
        "torch_dtype": torch.float16,              # or torch.bfloat16 on BF16-capable GPUs/CPUs
        "attn_implementation": "sdpa",           # try "flash_attention_2" only when model+CUDA stack supports it
    },
    tokenizer_kwargs={"model_max_length": max_seq_length},
)
model.max_seq_length = max_seq_length
embeddings = model.encode(
    texts_to_embed,
    batch_size=batch_size,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=normalize_embeddings,
    precision=encode_precision,
)
```

Notes:

- Keep `encode_precision="float32"` unless intentionally quantizing the returned vectors. In SentenceTransformers, `encode(precision=...)` controls output embedding precision/quantization choices such as int8/uint8/binary, not the same thing as loading model weights in fp16/bf16. Source: <https://www.sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.SentenceTransformer.encode>
- Use `model_kwargs={"torch_dtype":"float16"}` or `"bfloat16"` for PyTorch model weight/compute dtype. SentenceTransformers' efficiency guide explicitly lists torch-fp16 and torch-bf16 backends via `model_kwargs={"torch_dtype": "float16"}` and `model_kwargs={"torch_dtype": "bfloat16"}`. Source: <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#benchmarks>
- For attention backend experimentation, Hugging Face Transformers exposes `attn_implementation` values such as `sdpa` and `flash_attention_2`; PyTorch SDPA can choose optimized CUDA kernels, including FlashAttention-2 where supported. Sources: <https://huggingface.co/docs/transformers/en/attention_interface>, <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>

## Prioritized tuning plan

1. **Set and benchmark `batch_size` first.** Test 16/32/64/96/128 on a fixed 200-500 document slice, then confirm on the full 3,557-doc Q41 corpus. Raise until GPU memory is near saturation but below OOM. This is the lowest-risk change because it does not change embedding semantics.
2. **Set model dtype explicitly.** On CUDA, test `torch_dtype=float16`; on BF16-capable Ampere+ GPUs or supported CPUs, test `bfloat16`. Expect small numerical differences; validate retrieval metrics for the chosen dtype. SentenceTransformers reports fp16/bf16 and ONNX speedups can vary by text length/model, so use local corpus benchmarks instead of assuming a universal winner. Source: <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#recommendations>
3. **Cap token length to the retrieval strategy.** GTE-style configs are typically short-context; BGE-M3 can run to 8192 tokens, but BGE's own docs state that if such a long length is not needed, setting a smaller `max_length` speeds encoding. For SHRAG `3T+A`, start with 512 for GTE and test 512/1024 for BGE-M3 before allowing 8192. Sources: <https://bge-model.com/tutorial/1_Embedding/1.2.1.html>, <https://huggingface.co/BAAI/bge-m3>
4. **Length-bucket before encoding.** Sort or bucket documents by token count so each batch has similar lengths, then restore original order. Padding cost is batch-local; mixing very short titles with long abstracts wastes compute. This is especially important for BGE-M3 if any batches approach 1024+ tokens.
5. **Use dense-only BGE-M3 if BGE is retained.** BGE-M3 can return dense, sparse, and ColBERT-style vectors. For current SHRAG vector DB generation, request dense only (`return_dense=True`, `return_sparse=False`, `return_colbert_vecs=False`) if using `FlagEmbedding.BGEM3FlagModel`; do not compute sparse/ColBERT outputs unless the retrieval pipeline consumes them. Source: <https://bge-model.com/tutorial/1_Embedding/1.2.1.html>
6. **Try ONNX/OpenVINO after PyTorch knobs.** SentenceTransformers supports `backend="onnx"` and `backend="openvino"`; its guidance recommends ONNX-O4 for GPU short texts and OpenVINO/int8 variants for some CPU cases, but also warns to test with the specific model/data because longer texts can be slower than PyTorch. Sources: <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#onnx>, <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#openvino>, <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#recommendations>

## Model-specific guidance

### GTE (`Alibaba-NLP/gte-multilingual-base`)

- Treat GTE as the default speed baseline for this project because it already beat BGE-M3 by a wide margin in Exp12 cold embedding (about 99s vs 642s on the same 3,557-doc corpus).
- Start with `batch_size=64`, `torch_dtype=float16`, `max_seq_length=512`, `backend="torch"`, `attn_implementation="sdpa"`; compare against current default (`batch_size=32`, fp32/default dtype).
- If most SHRAG chunks are under ~500 characters, add an ONNX-O4 trial; SentenceTransformers' benchmark notes ONNX-O4 can be strong for short GPU inputs, while fp16/bf16 is recommended for longer GPU text. Source: <https://www.sbert.net/docs/sentence_transformer/usage/efficiency.html#recommendations>

### BGE-M3 (`BAAI/bge-m3`)

- Include BGE-M3 optimization only if practical: its 568M-parameter XLM-RoBERTa base and 8192-token capability are useful for long/multilingual/hybrid retrieval, but costly for dense-only indexing. BGE docs list BGE-M3 as 568M parameters, 2.27GB, with dense/sparse/multi-vector functions and 8192-token granularity. Source: <https://bge-model.com/tutorial/1_Embedding/1.2.1.html>
- If using SentenceTransformers, set `max_seq_length` explicitly; if using `FlagEmbedding.BGEM3FlagModel`, use `use_fp16=True`, pass an explicit `batch_size`, set `max_length` to the smallest value that preserves retrieval quality, and return dense vectors only. BGE/Hugging Face examples note `use_fp16=True` speeds computation with slight performance degradation and show `max_length=8192` as configurable. Sources: <https://huggingface.co/BAAI/bge-m3>, <https://bge-model.com/tutorial/1_Embedding/1.2.1.html>
- Practical first BGE trial: `use_fp16=True`, `batch_size=8 or 12`, `max_length=512`, dense-only. Then test `max_length=1024` if retrieval quality drops. Avoid 8192 for `3T+A` unless token histograms prove it is needed.

## Measurement checklist for Exp14 follow-up

For every trial, record:

- model, backend, dtype, batch size, max sequence length, attention implementation, normalization, and whether length bucketing was enabled;
- document count and token-length percentiles (`p50/p90/p95/p99/max`) before truncation;
- cold vs warm distinction: model load time, tokenization time if measured, encode time, vector DB write time, total wall time;
- GPU name, driver/CUDA/PyTorch/SentenceTransformers versions, and peak GPU memory;
- retrieval regression metrics for the selected candidate before replacing baseline embeddings.

## Recommended next code change

Introduce config parsing for `batch_size`, `torch_dtype`, `max_seq_length`, `backend`, `attn_implementation`, `normalize_embeddings`, and `length_bucket` in the encoder config loader, then pass those values into `generate_batch_embeddings()`. Keep defaults identical to current behavior (`batch_size=32`, fp32/default dtype, no explicit max length, no bucketing) so existing experiments remain reproducible unless the new fields are set.
