# Anime GAN
> A simple PyTorch Implementation of  Generative Adversarial Networks, focusing on anime face drawing.
This project use 3 type of GAN to experiment including DCGAN, WGAN_GP and DRAGAN.

## Dataset
Dataset I use is from this paper [Towards the Automatic Anime Characters Creation with Generative Adversarial Networks](https://arxiv.org/abs/1708.05509)

This dataset is collected by the authors which content 42000 face image of anime character.

You can download the training data from: https://drive.google.com/open?id=1bXXeEzARYWsvUwbW3SA0meulCR3nIhDb

## Generative Adversarial Networks (GANs)
*Name* | *Paper Link* | *Value Function*
:---: | :---: | :--- |
**DCGAN** | [Arxiv](https://arxiv.org/abs/1511.06434) | <img src = 'assets/DCGAN/DCGAN.png' height = '70px'>
**WGAN_GP**| [Arxiv](https://arxiv.org/abs/1704.00028) | <img src = 'assets/WGAN_GP/WGAN_GP.png' height = '70px'>
**DRAGAN**| [Arxiv](https://arxiv.org/abs/1705.07215) | <img src = 'assets/DRAGAN/DRAGAN.png' height = '70px'>

## Result for the dataset with fixed generation
All results are generated from the fixed noise vector.
*Name* | *Epoch 1* | *Epoch 50* | *Epoch 100* | *GIF*
:---: | :---: | :---: | :---: | :---: |
DCGAN | <img src = 'assets/DCGAN/DCGAN_epoch001.png' height = '200px'> | <img src = 'assets/DCGAN/DCGAN_epoch050.png' height = '200px'> | <img src = 'assets/DCGAN/DCGAN_epoch100.png' height = '200px'> | <img src = 'assets/DCGAN/DCGAN_generate_animation.gif' height = '200px'
WGAN_GP | <img src = 'assets/WGAN_GP/WGAN_GP_epoch001.png' height = '200px'> | <img src = 'assets/WGAN_GP/WGAN_GP_epoch050png' height = '200px'> | <img src = 'assets/WGAN_GP/WGAN_GP_epoch100.png' height = '200px'> | <img src = 'assets/WGAN_GP/WGAN_GP_generate_animation.gif' height = '200px'>
DRAGAN | <img src = 'assets/DRAGAN/DRAGAN_epoch001.png' height = '200px'> | <img src = 'assets/DRAGAN/DRAGAN_epoch050.png' height = '200px'> | <img src = 'assets/DRAGAN/DRAGAN_epoch100.png' height = '200px'> | <img src = 'assets/DRGAN/DRAGAN_generate_animation.gif' height = '200px'>

## Development Environment
Google Colab

## Usage
To reproduce the result of this project you download dataset from above link and place it in 'dataset/raw'.

Then for each of the model, run the bash file in 'script' folder.

Or, you can download the pretrained models from this link https://drive.google.com/drive/folders/1-3cKKUlq_jTdUUIhpHO4fzKBuGKxi03s?usp=sharing

## Web app
You can try to generate some faces using this app

https://anime-face-app.herokuapp.com/
