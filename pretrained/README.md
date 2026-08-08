# Pretrained models

This folder is where the pretrained weights go. The files themselves are **not**
tracked in git — download them from the
[GitHub Releases](https://github.com/ZHT150798/CorPiR/releases) page and place them here:

```
pretrained/
├── liif_edsr_pathology.pth.tar   # first-stage LIIF encoder (required for every scale)
├── corpir_x2_gen.pth
├── corpir_x4_gen.pth
└── corpir_x8_gen.pth
```

Only the `_gen.pth` files are needed for inference. The matching `_opt.pth` files
contain optimizer state and are only required to resume or fine-tune training.

See the "Pretrained Models" section of the main README for details.
