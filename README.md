# SSF for Efficient Model Tuning

This repo is the official implementation of our NeurIPS2022 paper "Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning" ([arXiv](https://arxiv.org/abs/2210.08823)). 




## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/dongzelian/SSF.git
cd SSF
```

- Create a conda virtual environment and activate it:

```bash
conda create -n ssf python=3.7 -y
conda activate ssf
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.6.5`:

```bash
pip install timm==0.6.5
```


- Install other requirements:

```bash
pip install -r requirements.txt
```


### Data preparation

- FGVC & vtab-1k

You can follow [VPT](https://github.com/KMnP/vpt) to download them. 

Since the original [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data) is processed with tensorflow scripts and the processing of some datasets is tricky, we also upload the extracted vtab-1k dataset in [onedrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liandz_shanghaitech_edu_cn/EnV6eYPVCPZKhbqi-WSJIO8BOcyQwDwRk6dAThqonQ1Ycw?e=J884Fp) for your convenience. You can download from here and then use them with our [vtab.py](https://github.com/dongzelian/SSF/blob/main/data/vtab.py) directly. (Note that the license is in [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data)).



- CIFAR-100
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

- For ImageNet-1K, download it from http://image-net.org/, and move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

- Robustness & OOD datasets

Prepare [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-R](https://github.com/hendrycks/imagenet-r) and [ImageNet-C](https://zenodo.org/record/2235448#.Y04cBOxByFw) for evaluation.



### Pre-trained model preparation

- For pre-trained ViT-B/16, Swin-B, and ConvNext-B models on ImageNet-21K, the model weights will be automatically downloaded when you fine-tune a pre-trained model via `SSF`. You can also manually download them from [ViT](https://github.com/google-research/vision_transformer),[Swin Transformer](https://github.com/microsoft/Swin-Transformer), and [ConvNext](https://github.com/facebookresearch/ConvNeXt).



- For pre-trained AS-MLP-B model on ImageNet-1K, you can manually download them from [AS-MLP](https://github.com/svip-lab/AS-MLP).



### Fine-tuning a pre-trained model via SSF

To fine-tune a pre-trained ViT model via `SSF` on CIFAR-100 or ImageNet-1K, run:

```bash
bash train_scripts/vit/cifar_100/train_ssf.sh
```
or 
```bash
bash train_scripts/vit/imagenet_1k/train_ssf.sh
```

You can also find the similar scripts for Swin, ConvNext, and AS-MLP models. You can easily reproduce our results. Enjoy!



### Robustness & OOD

To evaluate the performance of fine-tuned model via SSF on Robustness & OOD, run:

```bash
bash train_scripts/vit/imagenet_a(r, c)/eval_ssf.sh
```


### Citation
If this project is helpful for you, you can cite our paper:
```
@InProceedings{Lian_2022_SSF,
  title={Scaling \& Shifting Your Features: A New Baseline for Efficient Model Tuning},
  author={Lian, Dongze and Zhou, Daquan and Feng, Jiashi and Wang, Xinchao},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```


### Acknowledgement
The code is built upon [timm](https://github.com/rwightman/pytorch-image-models). The processing of the vtab-1k dataset refers to [vpt](https://github.com/KMnP/vpt), [vtab github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data), and [NOAH](https://github.com/ZhangYuanhan-AI/NOAH).
