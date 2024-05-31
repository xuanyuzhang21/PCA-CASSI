
<div align="center">
<h2>Progressive Content-aware Coded Hyperspectral Compressive Imaging
</h2>

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/abs/2303.09773)

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

Please download the CAVE dataset from [TSA-Net](https://github.com/mengziyi64/TSA-Net) and put it into the './datasets' directory.

```
python train.py --template pca --outf ./exp/pca/ --method pca
```

## Test

The model checkpoint file is provided in `./checkpoints`.
```
python test.py --template pca --outf ./exp/pca/ --method pca --pretrained_model_path ./checkpoints/model.pth
```
