# CorPiR

**Achieving High-Fidelity and Consistent Super-Resolution in Pathological Images using a Novel Consistency-Restricted Diffusion Framework**

International Journal of Computer Vision (IJCV) · DOI: [10.1007/s11263-026-02972-3](https://doi.org/10.1007/s11263-026-02972-3)

Official implementation of CorPiR, an effective two-stage framework for super-resolution pathological image reconstruction.


---

## Installation

The code was developed and tested with Python 3.10 and PyTorch 2.4.1 (CUDA 12.1)

```bash
git clone https://github.com/ZHT150798/CorPiR.git
cd CorPiR

conda env create -f environment.yml
conda activate corpir
```

`environment.yml` creates a Python 3.10 environment and installs everything listed in
`requirements.txt`. To manage the environment yourself instead:

```bash
conda create -n corpir python=3.10 -y
conda activate corpir
pip install -r requirements.txt
```

If you need a CUDA version other than 12.1, install the matching PyTorch wheel first
(see [pytorch.org](https://pytorch.org)) and then install the remaining dependencies:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch, model, data, core.logger; print(torch.__version__, torch.cuda.is_available())"
```

## Pretrained Models

Download the weights from the
[Releases](https://github.com/ZHT150798/CorPiR/releases) page and put them in `pretrained/`.

| File | Size | Needed for |
|---|---|---|
| `liif_edsr_pathology.pth.tar` | 25 MB | every scale (first-stage LIIF encoder) |
| `corpir_x2_gen.pth` | 423 MB | inference at ×2 |
| `corpir_x4_gen.pth` | 444 MB | inference at ×4 |
| `corpir_x8_gen.pth` | 527 MB | inference at ×8 |
| `corpir_x{2,4,8}_opt.pth` | 0.7–0.9 GB each | only to resume or fine-tune training |

```
pretrained/
├── liif_edsr_pathology.pth.tar
├── corpir_x2_gen.pth
├── corpir_x4_gen.pth
└── corpir_x8_gen.pth
```

Each diffusion checkpoint is tied to one up-sampling factor and the weights are not
interchangeable; `config/infer_CorPiR.json` picks the right file from its `scale` field.

## Data Preparation

Download `corpir_testset.zip` from the
[Releases](https://github.com/ZHT150798/CorPiR/releases) page and unpack it. It contains the
122 test patches used in the paper, and serves all three up-sampling factors:

```
/path/to/corpir_testset/
└── hr_128/
    ├── Bladder_10_10.png
    └── ...
```

A dataset is just a folder of high-resolution patches named `hr_<r_resolution>`; the
low-resolution input is derived from it at run time, so the same folder serves every
up-sampling factor. To train on your own images, put them in `hr_128/` the same way.

If your source images are not already 128×128, resize and crop them first with the
[SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) helper:

```bash
python data/prepare_data.py --path /path/to/source/images --out /path/to/your/trainset --size 32,128
```

`--size` is `<low>,<high>`, and the output folder gets a `_<low>_<high>` suffix, so the
command above writes `/path/to/your/trainset_32_128/hr_128/`. Only the `hr_<high>`
sub-folder is used.

## Inference

```bash
python infer.py -c config/infer_CorPiR.json
```

Before running, edit two fields in `config/infer_CorPiR.json`:

| Field | Meaning |
|---|---|
| `scale` | up-sampling factor: `2`, `4` or `8` |
| `datasets.val.dataroot` | path to the folder containing `hr_128`, e.g. `/path/to/corpir_testset` |

Everything else follows from `scale` — the low resolution, the attention resolution and
the checkpoint path are derived automatically, so a single config covers all three factors.

Results are written to `experiments/infer_CorPiR_X<scale>_<timestamp>/results/X<scale>/`,
one PNG per input image, keeping the original file names. Inference takes roughly 13 s per
128×128 image on an RTX 4090 (2000 denoising steps).

## Evaluation

```bash
python evaluate.py \
    --hr_dir  /path/to/corpir_testset/hr_128 \
    --sr_dir  experiments/infer_CorPiR_X4_<timestamp>/results/X4 \
    --scale   4 \
    --fid
```

Reports PSNR, SSIM, LPIPS and Consistency. `--fid` additionally computes FID and downloads
the InceptionV3 weights on first use. Images in the two folders are paired in sorted order,
and the script warns if the file names do not match.

## Training

```bash
python sr.py -p train -c config/train_CorPiR.json
```

Set `scale` and the two data roots in `config/train_CorPiR.json`:

```jsonc
"datasets": {
    "train": { "dataroot": "/path/to/your/trainset" },
    "val":   { "dataroot": "/path/to/your/valset" }
}
```

Training requires `pretrained/liif_edsr_pathology.pth.tar`. To resume, set `resume_state`
to a checkpoint prefix without the `_gen.pth` / `_opt.pth` suffix, e.g.
`experiments/train_CorPiR_X4_<timestamp>/checkpoint/latest`.

`consistency_constraint.last_steps` must be the same at training and inference time.

Checkpoints and logs go to `experiments/train_CorPiR_X<scale>_<timestamp>/`.

## Results

Reconstruction quality on the 122-image pan-cancer test set:

| Factor | PSNR ↑ | LPIPS ↓ | Consistency ↓ | FID ↓ |
|---|---|---|---|---|
| ×2 | 33.23 ± 5.97 | 0.048 ± 0.034 | 0.271 ± 0.091 | 23.0 |
| ×4 | 25.23 ± 4.85 | 0.173 ± 0.055 | 0.379 ± 0.124 | 49.6 |
| ×8 | 19.68 ± 3.68 | 0.304 ± 0.048 | 0.680 ± 0.190 | 63.8 |

`infer.py` fixes the random seed, so repeated runs produce identical images.

## Citation

Zhang, H., Zhang, X., Han, C. et al. Achieving High-Fidelity and Consistent Super-Resolution
in Pathological Images using a Novel Consistency-Restricted Diffusion Framework.
*Int J Comput Vis* **134**, 387 (2026). https://doi.org/10.1007/s11263-026-02972-3

```bibtex
@article{zhang2026corpir,
  title   = {{Achieving High-Fidelity and Consistent Super-Resolution in Pathological Images
              using a Novel Consistency-Restricted Diffusion Framework}},
  author  = {Zhang, Hongtai and Zhang, Xiuming and Han, Chu and Liu, Zaiyi and
             Song, Mingli and Madabhushi, Anant and Lu, Cheng},
  journal = {International Journal of Computer Vision},
  volume  = {134},
  number  = {8},
  pages   = {387},
  year    = {2026},
  doi     = {10.1007/s11263-026-02972-3}
}
```

## Acknowledgements

This implementation builds on
[SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). The
first-stage reconstruction and several image utilities are adapted from
[LIIF-PyTorch](https://github.com/Lornatang/LIIF-PyTorch), an implementation of
[LIIF](https://github.com/yinboc/liif) (Chen et al., CVPR 2021); both are released under the
Apache License 2.0 and the original copyright headers are kept in the corresponding files.

## License

Released under the [Apache License 2.0](LICENSE), the same license as the SR3 codebase this
work is derived from.
