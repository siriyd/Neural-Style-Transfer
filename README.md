# Neural Style Transfer

A PyTorch and Flask application for transferring the visual style of one image onto the content of another image. The project uses a pretrained VGG-based encoder, Adaptive Instance Normalization (AdaIN), and a learned decoder to produce stylized images through a web interface.

## Introduction

Neural style transfer separates an image into two complementary ideas:

- **Content**: the structure, objects, and spatial arrangement of an image.
- **Style**: the visual statistics, colors, textures, and patterns of another image.

This project extracts feature representations from both images, aligns the content features with the style statistics, and reconstructs the result with a trained decoder.

## About the Project

The application provides a browser-based interface where a user can:

1. Upload a content image.
2. Upload a style reference image.
3. Adjust the style strength using the alpha control.
4. Generate and download the stylized output.

The encoder is used only to extract features during inference. The decoder reconstructs an RGB image from the AdaIN-transformed feature representation.

## Tech Stack

- **Python**
- **PyTorch** and **Torchvision** for deep learning and image transforms
- **Flask** for the web application
- **Flask-WTF** and **WTForms** for the upload form
- **Pillow** for image loading and saving
- **Bootstrap**, CSS, and JavaScript for the interface
- **Gunicorn** for production serving

## CNN Core Concepts

### Convolutional Neural Networks

A convolutional neural network (CNN) learns visual patterns through layers of convolution filters. Early layers typically respond to simple features such as edges and colors, while deeper layers represent more complex structures, textures, and object-level patterns.

In this project, the VGG encoder contains convolution, ReLU, reflection-padding, and max-pooling layers. Pooling reduces spatial resolution while allowing deeper layers to capture broader visual context.

### Encoder and Decoder

- The **VGG encoder** converts an image into hierarchical feature maps.
- The **decoder** converts transformed feature maps back into an image.
- The VGG encoder is frozen during decoder training, so the decoder learns how to reconstruct stylized images from the feature space.

The encoder returns feature maps from four stages. During inference, the deepest feature representation is used for AdaIN transformation.

### Adaptive Instance Normalization

AdaIN transfers style statistics from the style feature map to the content feature map. For feature activations, it normalizes the content mean and standard deviation and then applies the style mean and standard deviation:

$$
\operatorname{AdaIN}(x, y) = \sigma(y)\left(\frac{x - \mu(x)}{\sigma(x) + \epsilon}\right) + \mu(y)
$$

where:

- $x$ is the content feature map.
- $y$ is the style feature map.
- $\mu$ is the channel-wise mean.
- $\sigma$ is the channel-wise standard deviation.
- $\epsilon$ prevents division by zero.

The result preserves the content structure while matching the style image's feature statistics.

### Style Strength

The application blends the transformed feature map with the original content feature map using alpha:

$$
 f = \alpha f_{style} + (1 - \alpha)f_{content}
$$

An alpha value of `1.0` applies the full style transformation. Lower values preserve more of the original content representation.

## Progressive Training Strategy

The decoder was trained using a two-stage progressive fine-tuning strategy to improve stability and final stylization quality.

### Stage 1 - Low-resolution training

- Final image size: **256 x 256**
- Epochs: **160**
- Style weight: **5**
- Content weight: **1**
- Learning rate: **1e-4**

The first stage trains the decoder at a lower resolution so it can learn the overall content-style transformation efficiently. At 256 x 256, the model focuses on large-scale structure, color distribution, and coarse texture patterns while requiring less computation and GPU memory. This provides a stable initialization before moving to higher-resolution images.

### Stage 2 - High-resolution fine-tuning

- Final image size: **512 x 512**
- Training is **resumed from the Stage 1 decoder and optimizer checkpoints**
- Style weight: **10**
- Content weight: **1**

In the second stage, the pretrained decoder is fine-tuned at 512 x 512 resolution rather than being initialized from scratch. The higher resolution exposes the network to finer spatial details such as edges, local textures, and brush-stroke patterns. Increasing the style weight from 5 to 10 gives greater importance to matching the style-image feature statistics, resulting in stronger stylization.

The overall idea is therefore:

> **256 x 256 training learns the coarse content-style mapping, while 512 x 512 fine-tuning refines high-resolution details and strengthens style transfer.**

The second stage reuses both the decoder weights and Adam optimizer state, preserving the learned parameters and optimization history from the first stage.

## Folder Structure

```text
NST/
├── app.py                         # Flask application and inference pipeline
├── train.py                       # Decoder training script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── vgg_normalised.pth             # Pretrained VGG encoder weights
├── content_data/                  # Content images used for training
├── style_data/                    # Style images used for training
├── content_imgs/                  # Additional content image resources
├── style_imgs/                    # Additional style image resources
├── examples/                      # Example images shown in the UI
├── experiment/
│   └── final_exp/
│       ├── decoder_final.pth      # Trained decoder checkpoint
│       └── options.txt            # Experiment settings
├── static/
│   └── uploads/                   # Uploaded and generated images
├── templates/
│   └── index.html                 # Web interface
└── utils/
    ├── models.py                  # VGG encoder and decoder definitions
    └── utils.py                   # Dataset, transforms, and AdaIN utilities
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The application requires the VGG encoder checkpoint at `vgg_normalised.pth` and the trained decoder checkpoint at `experiment/final_exp/decoder_final.pth`.

## How to Use

Start the Flask application from the project root:

```powershell
python app.py
```

Open the local application at:

```text
http://localhost:5000
```

Then upload a content image and a style image, choose the style strength, and select **Transfer Style**. The generated image can be downloaded from the result panel.

Supported image formats are `.png`, `.jpg`, and `.jpeg`.

## Training

The training script supports configurable image sizes, weights, learning rate, checkpoint saving, and checkpoint resumption.

Example Stage 1 command:

```powershell
python train.py --project-dir . --content_dir content_data --style_dir style_data --vgg vgg_normalised.pth --experiment stage1 --final_size 256 --content_size 256 --style_size 256 --epochs 160 --lr 1e-4 --content_weight 1 --style_weight 5
```

Example Stage 2 command:

```powershell
python train.py --project-dir . --content_dir content_data --style_dir style_data --vgg vgg_normalised.pth --experiment stage2 --final_size 512 --content_size 512 --style_size 512 --content_weight 1 --style_weight 10 --resume --decoder_path experiment/stage1/decoder_epoch_160.pth --optimizer_path experiment/stage1/optimizer_epoch_160.pth
```

Checkpoints and sample outputs are saved inside the selected experiment directory at the configured save interval.

## Inference Pipeline

```text
Content image + Style image
              |
              v
       Frozen VGG encoder
              |
              v
     Adaptive Instance Normalization
              |
              v
        Trained decoder
              |
              v
        Stylized image
```

## Notes

- CUDA is used automatically when available; otherwise the application runs on CPU.
- The decoder checkpoint is loaded with `map_location=device`, allowing it to run on a CPU-only machine even when the checkpoint was created on CUDA.
- Training is more practical on a CUDA-enabled GPU because of the memory and computation required by VGG feature extraction.
