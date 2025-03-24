# Joint Transformer and Mamba Fusion for Multispectral Object Detection
## Intro
Official Code for [Joint Transformer and Mamba Fusion for Multispectral Object Detection](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4932620).
## Installation
Install libraries including numpy, pytorch, timm, mamba-ssm, etc. according to `requirement.txt`

## Training
1. change [the pretrain-weight and data cfg](https://github.com/LiC2023/exp2/blob/ea71983afb9c5d097d8635df04ec6c43d6768419/train.py#L861).
2. change the [img2path](https://github.com/LiC2023/exp2/blob/ea71983afb9c5d097d8635df04ec6c43d6768419/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/exp2/blob/7b308d3739b58da3056491fafd6d8384438ba533/utils/dataloaders.py#L1484)
3. ```
   python train.py
   ```
   
## Test
1. change [the weights and data cfg](https://github.com/LiC2023/exp2/blob/0783358bd46486107450bcff59141cf9a337c4ee/val.py#L339)
2. change the [img2path](https://github.com/LiC2023/exp2/blob/ea71983afb9c5d097d8635df04ec6c43d6768419/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/exp2/blob/7a8c6f8de4735c263b9e690023ae60d203d7acdb/utils/dataloaders.py#L1506)
3. ```
   python val.py
   ```
   
## Visualize 
1. change [the weights and data cfg](https://github.com/LiC2023/exp2/blob/3cdd6aaf9aab58d47802eaabb94ff9ece409d462/result_vis.py#L16)
2. change the [img2path](https://github.com/LiC2023/exp2/blob/ea71983afb9c5d097d8635df04ec6c43d6768419/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/exp2/blob/7a8c6f8de4735c263b9e690023ae60d203d7acdb/utils/dataloaders.py#L1506)
3. ```
   python result_vis.py
   ```

## Dataset
The datasets and annotations used in this repo:   
-FLIR [[BaiDu Drive]](https://pan.baidu.com/s/1UGc_UgHM7fiKRiZGew_GIw) (code:jtm6)    
-LLVIP [[BaiDu Drive]](https://pan.baidu.com/s/1tFTxGCq40r-34Vhzhc5Njg) (code:jtm6)     
-M<sup>3</sup>FD [[BaiDu Drive]](https://pan.baidu.com/s/1EDdwiANvKgvTXGB3X-z8bw)(code:jtm6)    

## Weight
-FLIR [[BaiDu Drive]](https://pan.baidu.com/s/1JSndBVsRphFcdM21BK0nHg) (code:jtm6)    
-LLVIP [[BaiDu Drive]](https://pan.baidu.com/s/1-OsdF8x7ZL3TFVfkn-6EwA) (code:jtm6)    
-M<sup>3</sup>FD [[BaiDu Drive]](https://pan.baidu.com/s/1PHmRPWtQmxYSwEc_K90D7g) (code:jtm6)    

## Cite
If you find our model/method/dataset useful, please cite our work:
```
@article{li2025joint,
  title={Joint Transformer and Mamba fusion for multispectral object detection},
  author={Li, Chao and Peng, Xiaoming},
  journal={Image and Vision Computing},
  pages={105468},
  year={2025},
  publisher={Elsevier}
}
```
   
