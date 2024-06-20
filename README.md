# Text_to_Image-
This repository contains the implementation of a Text-to-Image generation model that leverages the power of StyleGAN for image synthesis and a Transformer-based encoder for processing textual descriptions. The goal of this project is to generate high-quality images from textual descriptions.
Table of Contents
  Introduction
  Features
  Architecture
    1) Generator
    2) Discriminator
    3) Transformer Encoder
Dataset Preparation
Training
Results
Requirements
Usage
Acknowledgments


Introduction

Text-to-Image generation is a challenging task that involves generating visually realistic images from given textual descriptions. This repository implements a model that combines the strengths of StyleGAN for high-quality image generation and Transformer models for capturing the semantic content of text descriptions.


Features
   1) High-quality image generation using StyleGAN.
   2) Semantic text encoding using Transformer-based models.
   3) Gradient penalty for stabilizing GAN training.
   4) Progressive growing GAN for handling high-resolution images.
   5) Custom dataset handling for paired text and image data.
 

![Screenshot (1060)](https://github.com/Dasuional/Text_to_Image-/assets/103253038/82273e5b-df40-4145-b3db-348daa6398d4)


Architecture
 Generator
    The Generator uses StyleGAN architecture, modified to incorporate textual information into the generation process. The input to the generator includes noise 
    vectors concatenated with encoded text vectors.

 Discriminator
    The Discriminator, also based on the StyleGAN architecture, is designed to distinguish between real and generated images. It processes both the images and 
    their corresponding text encodings.

Transformer Encoder
    The Transformer Encoder converts text descriptions into dense vector representations. It uses multi-head self-attention mechanisms and positional encoding to 
    capture the semantic meaning of the text.

Dataset Preparation
     The dataset should consist of images and their corresponding textual descriptions. Images are stored in the directory specified by image_dir, and text files 
     containing descriptions are stored in label_dir. Text files should be preprocessed into dense vectors using the Transformer encoder.

Training
  Training involves alternating between updating the generator and discriminator. The training script includes steps for:

    1) Loading and preprocessing the dataset.
    2) Computing the gradient penalty for the discriminator.
    3) Progressive growing of the GAN, starting from low-resolution images and progressively increasing the resolution.
    4) Tracking and displaying training progress.

Results
  Results include generated images that correspond to given textual descriptions. Sample outputs can be visualized using scripts provided in the repository.

Requirements
    1) Python 3.8+
    2) PyTorch
    3) torchvision
    4) numpy
    5) matplotlib
    6) tqdm

Usage
Dataset Preparation
  1) Place images in the directory specified by image_dir.
  2) Place text files with descriptions in the directory specified by label_dir.

Training
   1) Initialize and preprocess the dataset
     from dataset import ImageLabelDataset
     dataset = ImageLabelDataset(image_dir, label_dir, transform, encode_label_fn)
     loader = get_loader(dataset)
   
   2) Train the model
      from train import train_fn
      alpha = train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen)

Generating Images
      1) Generate images from new text descriptions using the trained generator.

Acknowledgments
    1)This repository is inspired by the StyleGAN and Transformer models.
    2)Special thanks to the authors of some articles and blogs who contributed to these models.
