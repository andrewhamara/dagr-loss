# Setup: N-Caltech101 Training on CUDA 12.4+ Servers

This guide covers setting up DAGR for N-Caltech101 training on a cloud server with
CUDA 12.4+ drivers. The repo requires PyTorch 1.11.0 + CUDA 11.3, but CUDA drivers
are forward-compatible—a 12.4 driver can execute code compiled with CUDA 11.3. The
key is installing `nvcc` 11.3 via conda so the CUDA extensions compile against the
correct toolkit version.

## Full Setup

```bash
# 1. Clone repo
WORK_DIR=/path/to/work/directory
cd $WORK_DIR
git clone git@github.com:uzh-rpg/dagr.git
DAGR_DIR=$WORK_DIR/dagr
cd $DAGR_DIR

# 2. Create conda env + install PyTorch 1.11.0 with CUDA 11.3
conda create -y -n dagr python=3.8
conda activate dagr
conda install -y setuptools==69.5.1 mkl==2024.0 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# 3. Install nvcc 11.3 inside the conda env so CUDA extensions compile correctly.
#    Without this, setup.py finds the system nvcc (12.4) which is incompatible
#    with PyTorch's 11.3 runtime and the build fails.
conda install -y -c conda-forge cudatoolkit-dev=11.3

# 4. Point CUDA_HOME to conda's toolkit BEFORE any compilation steps.
#    Consider adding these to ~/.bashrc or a setup script.
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# Verify nvcc is the conda one
which nvcc          # should print $CONDA_PREFIX/bin/nvcc
nvcc --version      # should show 11.3

# 5. Install pytorch-geometric stack (auto-detects torch 1.11.0 + cu113)
bash install_env.sh

# 6. Install library dependencies (detectron2, YOLOX, dsec-det)
#    NOTE: This script clones repos via git@github.com:... — you need SSH keys
#    set up, or manually edit the script to use HTTPS URLs.
bash download_and_install_dependencies.sh
conda install -y h5py blosc-hdf5-plugin

# 7. Install dagr package (compiles asy_tools and ev_graph_cuda extensions)
pip install -e .

# 8. Download N-Caltech101 dataset
wget https://download.ifi.uzh.ch/rpg/dagr/data/ncaltech101.zip -P $DAGR_DIR/data/
cd $DAGR_DIR/data/ && unzip ncaltech101.zip && rm ncaltech101.zip
cd $DAGR_DIR

# 9. Train
WANDB_MODE=disabled python scripts/train_ncaltech101.py \
    --config config/dagr-l-ncaltech.yaml \
    --exp_name ncaltech_l \
    --dataset_directory $DAGR_DIR/data/ \
    --output_directory $DAGR_DIR/logs/ \
    --l_r 0.01
```

## Verification

Run these after step 7 to confirm everything is working:

```bash
# Check PyTorch + CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 1.11.0 11.3 True

# Check CUDA extensions compiled
python -c "import asy_tools; print('asy_tools OK')"
python -c "import ev_graph_cuda; print('ev_graph_cuda OK')"

# Check full import chain
python -c "from dagr.model.networks.dagr import DAGR; print('DAGR import OK')"
```

Then start training (step 9) and confirm loss is decreasing in the first epoch's
tqdm output.

## Key Details

- **Why this works**: CUDA drivers are forward-compatible. A 12.4 driver can execute
  code compiled with CUDA 11.3. The failure scenario is when the *system* `nvcc`
  (12.4) compiles extensions against PyTorch's 11.3 runtime—the version mismatch
  causes build errors. Using conda's `nvcc` 11.3 keeps everything consistent.

- **CUDA_HOME must be set before steps 5–7**. If you forget, `pip install -e .` will
  find the system nvcc 12.4 and fail. Add the exports to `~/.bashrc` or run them
  every time you activate the env.

- **SSH keys**: `download_and_install_dependencies.sh` clones three repos via SSH
  (`git@github.com:...`). Either set up GitHub SSH keys or edit the script to replace
  `git@github.com:` with `https://github.com/`.

- **Checkpoints**: `last_model.pth` is saved every epoch; `best_model_mAP_*.pth` is
  saved every 3 epochs when mAP improves. With wandb disabled, output goes to
  `$DAGR_DIR/logs/ncaltech101/detection/None/`.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `nvcc fatal: Unsupported gpu architecture 'compute_XX'` | System nvcc (12.4) being used instead of conda's | Re-export `CUDA_HOME=$CONDA_PREFIX` and `PATH=$CONDA_PREFIX/bin:$PATH` |
| `undefined symbol` errors when importing `asy_tools` or `ev_graph_cuda` | Extensions compiled with wrong CUDA version | `pip install -e .` again after fixing CUDA_HOME |
| `torch-scatter` install fails | PyG wheel URL mismatch | Check `install_env.sh` constructs URL with `cu113`; run `python -c "import torch; print(torch.__version__, torch.version.cuda)"` |
| SSH clone failures in `download_and_install_dependencies.sh` | No GitHub SSH key | Add SSH key or edit script to use HTTPS URLs |
| `ModuleNotFoundError: No module named 'dagr'` | `pip install -e .` not run or failed silently | Re-run `pip install -e .` and check for errors |
