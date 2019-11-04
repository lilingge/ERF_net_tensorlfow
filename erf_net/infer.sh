#!/bin/bash
CUDA_VISIBLE_DEVICE=1 python infer.py --model=erf --ckpt_path=/data1/llg/Documents/erf_net/log/  --images_path=/data1/llg/Documents/cityscapes/trainImages.txt --output_path=/data1/llg/Documents/erf_net/segmentation --mask_path=./model_inference/erf_mask
