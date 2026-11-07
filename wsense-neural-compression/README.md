# WSENSE Underwater Neural Compression Challenge – 🥇 1st place

**Kaggle competition, WSENSE training camp at Sapienza University of Rome – September 2026**
Team of 3: Alessandro Querqui, [TEAMMATE 2], [TEAMMATE 3]

**Result:** 1st place out of 27 teams, final private leaderboard score **1.2446** (the organisers' reference is 1.00).

## The challenge

Build learned image codecs for underwater imagery (useful for transmitting images over very low-bandwidth underwater links). The competition had three tasks:

1. **Learned compression:** pick one operating point with the best trade-off between rate and quality.
2. **Fixed byte budget:** every image must fit under the same byte budget; an image over budget scores as if it was never compressed.
3. **Same-scene groups:** images from the same scene are compressed together, so information can be shared across them.

Each task is scored by combining PSNR, MS-SSIM and compression ratio, each normalised by a reference value; the final score is the mean over the three tasks. A score above 1 means beating the reference.

## Our approach

We did not design new architectures: we built a selection pipeline on top of strong pretrained neural codecs and adapted them to the underwater domain.

- **Portfolio of pretrained codecs:** CompressAI `mbt2018-mean` (MSE and MS-SSIM checkpoints), CompressAI `ssf2020` (neural video compression) and the public HPCM-1B and HPCM-Large models.
- **Real encode/decode search:** for each image we actually encode and decode every candidate configuration (model, checkpoint, input scale, quantisation step) and keep the best one by measured quality and size: 54 configurations for Task 1, 85 for Task 2 and 13 for Task 3.
- **Byte budget (Task 2):** candidates over the budget, header included, are discarded before ranking.
- **Domain adaptation:** we fine-tuned only the decoder of two CompressAI models on underwater images (loss: 100·MSE + (1 − MS-SSIM)), keeping encoder and entropy model frozen.
- **Scene redundancy (Task 3):** two strategies compared per scene: neural video coding (SSF2020) on the ordered frames, or a shared reference image (the scene mean) compressed once, plus small residual grids per image.
- **Reproducibility:** fixed seed, deterministic GPU algorithms and a fully offline pipeline (dependencies and weights attached as Kaggle datasets, checked with SHA-256).

## Files

- `[notebook_name].ipynb` – commented submission notebook
- `[notebook_name].pdf` – PDF version of the commented notebook

## Tools

Python, PyTorch, CompressAI, NumPy, Kaggle (GPU)
