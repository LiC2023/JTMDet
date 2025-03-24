# Joint Transformer and Mamba Fusion for Multispectral Object Detection
## Intro
Official Code for [Joint Transformer and Mamba Fusion for Multispectral Object Detection](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4932620).
## Installation
Install libraries including numpy, pytorch, timm, mamba-ssm, etc. according to `requirement.txt`

## Training
1. change [the pretrain-weight and data cfg](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/train.py#L861).
2. change the [img2path](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L1506)
3. ```
   python train.py
   ```
   
## Test
1. change [the weights and data cfg](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/val.py#L339)
2. change the [img2path](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L1506)
3. ```
   python val.py
   ```
   
## Visualize 
1. change [the weights and data cfg](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/result_vis.py#L16)
2. change the [img2path](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L425) and [filter label](https://github.com/LiC2023/JTMDet/blob/3325917b71a8c9b4224d5ebcdde650a6b38091d8/utils/dataloaders.py#L1506)
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

## Pretrain_YOLOv5_Weight
-YOLOv5 [[BaiDu Drive]](https://pan.baidu.com/s/19UvONHvD4F67oks2zNal4w) (code:jtm6)

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
   
