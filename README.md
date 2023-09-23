# SSF for Efficient Model Tuning

This branch is the duplicated version of the official implementation of the NeurIPS2022 paper "Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning" ([arXiv](https://arxiv.org/abs/2210.08823)). It was built by [dragonbra](https://github.com/dragonbra) and [dercaft](https://github.com/dercaft) to consider more functions. The owner of SSF only checked part of the code.


## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/dercaft/SSF.git
cd SSF
```

- Create a conda virtual environment and activate it:

```bash
conda create -n ssf -f ssfn.yaml -y
conda activate ssf
```

### Data preparation
在项目路径下，新建DATA_PATH.sh:
```bash
touch DATA_PATH.sh
```
DATA_PATH.sh内应是数据集路径以及要输出的log,checkpoint路径
```bash
export VTAB_PATH="<dataset_abspath>"
export OUTPUT_PATH="<output_abspath>"

```

每个新建的训练脚本sh文件，开头都要 `source DATA_PATH.sh`!!!

MAC203服务器上有数据集，可以直接使用，不需要下载。

数据集路径: /data/VTAB/vtab-1k

输出路径为: ${OUTPUT_PATH}

MAC235服务器上有数据集，可以直接使用，不需要下载。

数据集路径: /media/disk1/VTAB/vtab-1k

输出路径为: /media/disk1/wyh/SSF

如果要自己下载，按以下步骤：

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

### Log 注意事项：

```python
  _logger.info(
      '{0}: [{1:>4d}/{2}]  '
      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
      'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
      'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
      'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
          log_name, batch_idx, last_idx, batch_time=batch_time_m,
          loss=losses_m, top1=top1_m, top5=top5_m))
```
这里每个 Acc@1/5: <log输出时，最后一个batch的准确率> (<整体平均的准确率>)

所以会出现前面acc都高，但是括号里的acc比前面都低的情况。

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
