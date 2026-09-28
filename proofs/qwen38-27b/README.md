# Qwen3.8-27B release evidence

- `bit_exact_receipt.json`: independent CPU verification of all 1,199
  tensors, 10 sidecars, parent Hugging Face LFS hashes, and bundle shard
  hashes. The original full receipt SHA-256 is recorded inside.
- `FASTSESSION_RESULT.json`: same RTX PRO 6000 installed-wheel comparison of
  TBE and BF16, with links to sanitized raw text/image and MTP receipts in
  this directory. It records the tested wheel and tune hashes and the scope
  limits. The original private GCS summary SHA-256 is recorded inside.

The model's encoded weight shards are distributed through
[Hugging Face](https://huggingface.co/Tsotchke-Corporation/Qwen3.8-27B-GeoRefine-TBE),
not this GitHub repository.
