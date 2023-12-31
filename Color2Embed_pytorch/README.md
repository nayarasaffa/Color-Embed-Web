[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xyq-kuTWzvoQH4r7d5C7YN7sVe19pUv0)
# Color2Embed_pytorch
This is the 3d-part realisation of **Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings** (https://arxiv.org/abs/2106.08017) on PyTorch

# Disclaimer
The article has several ambiguous points that I decided as best I could, in addition, I moved away from the things clearly described in the article: I changed the color encoder to resnet18 instead of simple network in the article and increased the learning rate by 10 times (from 10-4 to 10-3), btw results look like very similar with authors results.

### Abstract
In this paper, we present a fast exemplar-based image colorization approach using color embeddings named Color2Embed. Generally, due to the difficulty of obtaining input and ground truth image pairs, it is hard to train a exemplar-based colorization model with unsupervised and unpaired training manner. Current algorithms usually strive to achieve two procedures: i) retrieving a large number of reference images with high similarity for preparing training dataset, which is inevitably time-consuming and tedious; ii) designing complicated modules to transfer the colors of the reference image to the target image, by calculating and leveraging the deep semantic correspondence between them (e.g., non-local operation), which is computationally expensive during testing. Contrary to the previous methods, we adopt a self-augmented self-reference learning scheme, where the reference image is generated by graphical transformations from the original colorful one whereby the training can be formulated in a paired manner. Second, in order to reduce the process time, our method explicitly extracts the color embeddings and exploits a progressive style feature Transformation network, which injects the color embeddings into the reconstruction of the final image. Such design is much more lightweight and intelligible, achieving appealing performance with fast processing speed.

# Train
0. Setup enviroment, my (a little redundant) conda enviroment in `environment.yml`
1. Download ImageNet part from [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
2. Unpack it and made train list: each line of simple text file should be path to file, e.g. `n09428293/n09428293_23938.JPEG` or full path, set up path to this text file in `config.IMAGENET_LIST`. If you have full path in list, set `config.IMAGENET_PREFIX = ''` or to prefix if you extract it, btw `os.path.join(config.IMAGENET_PREFIX, <image_list_item>)` should be valid path to image
3. Change config batch size for your GPU (I have V100 and batch 32, it is around 20 GB GPU memory)
4. Run training with DDP from `color2embed` folder `python -m torch.distributed.launch --nproc_per_node=8 train.py`, where 8 - number of GPUs

Code samples some train parts are in `color2embed/test_parts.ipynb`

My train log:
![Train log](/readme_images/train_log.PNG)
# Results on train in the end of training
Target:

![Target](/readme_images/last_targets.png)
Color source:

![Color source](/readme_images/last_color_sources.png)
Ground truth: 

![Ground truth](/readme_images/last_gt.png)
Generated images:

![Generated images](/readme_images/last_predicted.png)
Predicted "a" channel:

![Predicted "a" channel](/readme_images/a_channel.png) 
Predicted "b" channel:

![Predicted "b" channel](/readme_images/b_channel.png)
# Result on test images
Good samples:

![good](/result_images/good1.png)
![good](/result_images/good2.png)
![good](/result_images/good3.png)
![good](/result_images/good4.png)
![good](/result_images/good5.png)

Bad samples:

![bad](/result_images/bad1.png)
![bad](/result_images/bad2.png)
# Inference
Download weights from https://drive.google.com/file/d/1xmn-8FvKqm6MoSVYYQq9Rn-BdnOTdrwx/view?usp=sharing and put it in `trained_model` folder

See `color2embed/inference.ipynb` for details

# References:
* Color2Embed: Fast Exemplar-Based Image Colorization using Color Embeddings: https://arxiv.org/abs/2106.08017
* TSP augmentation: https://github.com/cheind/py-thin-plate-spline
* UNet parts: https://github.com/milesial/Pytorch-UNet
* Modulated Convolution: https://github.com/rosinality/stylegan2-pytorch
