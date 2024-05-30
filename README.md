
<div align="center">
<h3>Progressive Content-aware Coded Hyperspectral Compressive Imaging
</h3>


[![ArXiv](https://img.shields.io/badge/IEEE_Xplore-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.09773)

[Xuanyu Zhang](https://xuanyuzhang21.github.io/), [Bin Chen](https://scholar.google.com/citations?user=aZDNm98AAAAJ&hl=zh-CN&oi=ao), Wenzhen Zou, Shuai Liu, Yongbing Zhang, Ruiqin Xiong and  [Jian Zhang](https://jianzhang.tech/)

*School of Electronic and Computer Engineering, Peking University*
</div>

Accepted for publication as a Regular paper in the IEEE Transactions on Circuits and Systems for Video Technology (TCSVT).

## Core Ideas
![idea](./asserts/PCA-CASSI.png)


## Environment

```shell
pip install -r requirements.txt
```
## Train
Download the dataset of [Waterloo Exploration Database](https://kedema.org/project/exploration/index.html) and put all images in the `pristine_images` directory (containing 4744 `.bmp` image files) into `./data/train`, then run:

```
python train.py --template pca --outf ./exp/pca/ --method pca
```

## Test

The model checkpoint file is provided in `./checkpoints`.
```
python test.py --template pca --outf ./exp/pca/ --method pca --pretrained_model_path ./checkpoints/model.pth
```
