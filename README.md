# GVHMR-Masked

> This repository extends [GVHMR](https://github.com/zju3dv/GVHMR) with a masked generative local-pose recovery branch.
> For the original GVHMR documentation, please refer to the [original GVHMR README](README_GVHMR.md).

**Status**

- Current checkpoint: `step32000.ckpt` (32,000 / 60,000 steps)

**Upcoming**

- [ ] 60,000-step checkpoint verification
- [ ] `F_smoother` fine-tuning

## Install

### Environment

Ubuntu 22.04 (WSL2)

```bash
git clone https://github.com/37g55555/GVHMR.git
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install "git+https://github.com/mattloper/chumpy.git" --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

### Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints
```

1. You need to sign up for downloading SMPL(https://smpl.is.tue.mpg.de/) and SMPLX(https://smpl-x.is.tue.mpg.de/) or [Body Models](https://livecauac-my.sharepoint.com/:f:/g/personal/rlawldls379_cau_ac_kr/IgChhpmSCZdkRLkuIbuCqI0LAfR9NAKnN0KbKOP0rWMlI4g?e=FapXcp). And the checkpoints should be placed in the following structure:

```text
inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPL-X (prediction and evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)
```

2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

```text
inputs/checkpoints/
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

3. Download the [GVHMR-Masked checkpoints](https://livecauac-my.sharepoint.com/:f:/g/personal/rlawldls379_cau_ac_kr/IgBYdj2JJhFyTY8Pr3dVHRE6AVMKHaUkdDBeQYN7AG4LAic?e=6nyDjj) and place them in the following structure:

```text
inputs/checkpoints/
├── gvhmr_masked/
│   └── step32000.ckpt
└── tokenhmr/
    └── tokenizer.pth
```

**Data**

You can download them from [Google Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```

## Demo

Demo entry points are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.

```bash
# Add --masked_ckpt_path to run the demo with a trained masked-pose checkpoint.
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s \
	--masked_ckpt_path=inputs/checkpoints/gvhmr_masked/step32000.ckpt \
  --output_root=outputs/demo_masked
```

### Optional: Visualization with Viser

Download the Viser visualization tool from [OneDrive](https://livecauac-my.sharepoint.com/:f:/g/personal/rlawldls379_cau_ac_kr/IgABOkQ_SweJTbAo7slGJ4UVAauhVuykKjpflBtN9P0XpSg?e=Sf3cda).

```bash
conda create -n viser python=3.11 -y
conda activate viser

pip install viser pandas

python pose_viser.py
```

By default, the demo exports CSV files to `../data/gvhmr_res` when launched from the GVHMR project root. Upload an exported CSV file to the Viser interface to inspect the reconstructed motion interactively. Use the "Overlay" button to display multiple loaded motions in the same scene for direct comparison.

## Reproduce

**Test:** To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:

```bash
python tools/train.py \
    global/task=gvhmr/test_3dpw_emdb_rich \
    exp=gvhmr/masked_pose/tokenhmr_moro \
    ckpt_path=inputs/checkpoints/gvhmr_masked/step32000.ckpt
```

To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

**Train:** To train the model, use the following command:

```bash
# The `step32000.ckpt` checkpoint was trained with 1x RTX 5060 Ti for 32,000 steps using a batch size of 1. Note that different GPU and training settings may lead to different results.
python tools/train.py exp=gvhmr/masked_pose/tokenhmr_moro
```

During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).
