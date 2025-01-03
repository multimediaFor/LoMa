## Image Forgery Localization with State Space Models
An official implementation code for paper "[Image Forgery Localization with State Space Models.](https://arxiv.org/abs/2412.11214)". This repo provide codes and trained weights.

## Framework
<p align='center'>  
  <img src='./images/LoMa.png' width='900'/>
</p>

## Dependency
- torch 1.13.1+cu117
- torchvision 0.14.1+cu117
- python 3.10
- causal-conv1d 1.0.0
- mamba-ssm 1.0.1
- selective_scan 0.0.2

## Usage

For example to test:
download [LoMa.pth](https://www.123684.com/s/2pf9-ucWHv)
```bash
cd LoMa/models
python generate_npy.py
python test.py 
```

## Citation
If you use this code for your research, please cite our paper
```
@article{lou2024image,
  title={Image Forgery Localization with State Space Models},
  author={Lou, Zijie and Cao, Gang},
  journal={arXiv preprint arXiv:2412.11214},
  year={2024}
}
```
## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
