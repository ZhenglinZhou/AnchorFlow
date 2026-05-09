# AnchorFlow: Training-Free 3D Editing via Latent Anchor-Aligned Flows

<a href='https://zhenglinzhou.github.io/AnchorFlow/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2511.22357'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/datasets/chengzgui/Eval3DEdit'><img src='https://img.shields.io/badge/Dataset-HuggingFace-yellow'></a>

This repo provides the official implementation of **AnchorFlow**, a **training-free** framework for 3D shape editing. The method performs editing directly in the 3D latent space by aligning source and target flow trajectories with latent anchors, enabling semantic-consistent, identity-preserving, and **mask-free** 3D editing across both rigid and non-rigid scenarios.

![teaser](./assets/teaser.png)

## Installation

AnchorFlow builds upon [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), which serves as the base flow model. We recommend using CUDA 12.4 (as suggested in the official Hunyuan3D instructions) or CUDA 12.1.

```bash
# Create a conda environment
conda create -n anchorflow python=3.10
```

* CUDA=12.4

```bash
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
--index-url https://download.pytorch.org/whl/cu124
# Install dependencies
pip install -r requirements_cuda124.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

* CUDA=12.1 (also support CUDA=12.2)

```bash
# Install PyTorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
--index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cuda121.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Usage

Run AnchorFlow with default settings. More examples can be found in `./examples`.

```bash
python3 src/anchorflow.py
```

## Prepare Editing Data

Given a `source shape` and `editing prompt`, we first construct the editing conditions, including the `source image` and `target image`. Then, we organize the inputs and perform 3D editing.

* Step1: Rendering the `source image` from the `source shape`.
  Use the **Render Multiview Images** script in [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/DATASET.md). Then select one suitable rendering as the `source image`.
* Step2: Construct the `target image` using a 2D editing model: Apply a 2D editing model (e.g., [Nano Banana](https://aistudio.google.com/models/gemini-2-5-flash-image)) to edit the `source image` according to the given `editing prompt`, producing the `target image`.
* Step3: Organize the inputs and perform 3D editing. Place all required inputs in a folder and run the 3D editing pipeline. The process requires the following files:
  * `src.glb`: the `source shape`
  * `src.png`: the `source image`
  * `edited.png`: the `target image`

You can find example setups in `./examples`.

The **Eval3DEdit benchmark** is available [here](https://huggingface.co/datasets/chengzgui/Eval3DEdit).

## Notes

- If you have questions or find bugs, feel free to open an issue or email the first author (zhenglinzhou@zju.edu.cn).
- If you encouter `EGL: cannot open shared object file: No such file or directory` error during rendering mesh, try to install following packages: `sudo apt-get install libegl1-mesa libgl1-mesa-glx`.

## Acknowledgements

Our repo is built on top of several several awesome projects and works, including [FlowEdit](https://github.com/fallenshock/FlowEdit), [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) and [EditP23](https://github.com/editp23/editp23).

## Cite

If you find AnchorFlow useful for your research and applications, please cite us using this BibTex:

```bibtex
@article{zhou2025anchorflow,
  title={AnchorFlow: Training-Free 3D Editing via Latent Anchor-Aligned Flows},
  author={Zhou, Zhenglin and Ma, Fan and Gui, Chengzhuo and Xia, Xiaobo and Fan, Hehe and Yang, Yi and Chua, Tat-Seng},
  journal={arXiv preprint arXiv:2511.22357},
  year={2025},
}
```

