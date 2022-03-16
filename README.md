# TxST (Arbitrary text driven artistic style transfer)
text-driven image style transfer (TxST) that leverages advanced image-text encoders to control arbitrary style transfer

By Zhi-Song Liu, Li-Wen Wang, Wan-Chi Siu and Vicky Kalogeiton

This repo only provides simple testing codes and pretrained models.

Please check our [paper](https://arxiv.org/pdf/2202.13562.pdf)

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch >= 1.8
    NVIDIA GPU + CUDA >=11.0
    
# Usage
## First, install additional dependencies by running
```sh
$ pip install -r requirements.txt
```
## Second, install [CLIP](https://github.com/openai/CLIP)
```sh
$ pip install git+https://github.com/openai/CLIP.git
```

## Third, update the CLIP model file
find the "model.py" at the prebuilt CLIP lib path, if you are using Conda, you can find it at 
```sh
$ miniconda3/envs/conda_env_name(#your conda-env name#)/lib/python(#your version#)/site-packages/clip
```
then replace it with the new [file](https://drive.google.com/file/d/1h-Wh6tUGf9OTrGkJSAyvZRymTfQXc--O/view?usp=sharing)

# Testing
## artist style transfer using reference images
put your content images under "data/content" and put your style images under "data/style"

then run,
```sh
$ python eval_ST_img.py
```

## artist style transfer using texts
run
```sh
$ python demo_edit_art_style.py --content #path to your content image --style #artistic text
```

## texture style transfer using texts
run
```sh
$ python demo_edit_texture_style.py --content #path to your content image --style #texture text
```

# Visualization
Here we show some cases on Wikiart style transfer using just texts as style description.
![figure1](/figure/figure1.png)
![figure2](/figure/figure2.png)
