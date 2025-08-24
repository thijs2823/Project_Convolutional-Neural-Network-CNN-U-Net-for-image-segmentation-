# Project Convolutional Neural Network (CNN) / U-Net for image segmentation

My first attemt in making an Convolutional Neural Network (for educational purpuses), in which I use U-Net for training on retina and mask imagings. 

## UNet Segmentation on STARE Dataset

This repository contains a PyTorch implementation of a UNet model for biomedical image segmentation using the STARE dataset [1,2]. The dataset consists of retina images with annotations from two different annotators. This implementation handles both annotators’ masks to improve training robustness and generalization.

## Features

* UNet architecture with BatchNorm and ReLU activations.
* Dual loss function: **Focal Loss** + **Tversky Loss**.
* Data augmentation: random horizontal/vertical flips, brightness and contrast adjustments.
* Support for images with multiple annotators.
* Cropping and resizing to a consistent input size.
* Training and validation loops with logging and early stopping.
* Training loss and validation loss plotted and saved as `training_loss_curve.png`.

## Requirements

* Python 3.10+
* PyTorch 2.x
* torchvision
* numpy
* matplotlib
* PIL
* scikit-learn
* tqdm

Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-learn tqdm
```

## Dataset Setup

The code expects the STARE dataset structured as:

```
STARE/
├── stare_images/        # .ppm.gz images
├── stare_labels-a/      # First annotator masks (.ah.ppm.gz)
└── stare_labels-b/      # Second annotator masks (.vk.ppm.gz)
```

Update the paths in the `Config` class accordingly.

## Usage

* Training and validation split is 80/20.
* Uses GPU if available.
* Training will stop early if validation Dice score does not improve for 10 consecutive epochs.


## How Multiple Masks Are Used

Each image may have two masks from different annotators. During training:
* Each mask is treated as a separate training example; this increases effective training data and improves robustness to annotation differences.

## Output

* Console logs of training progress and validation scores.
* `training_loss_curve.png`: plots training and validation loss over epochs.
* The trained model can be saved manually by uncommenting the `torch.save()` line inside the best validation Dice check.

Trainings data:
## Training Progress

The following plots show the training and validation performance over 60 epochs.

<img width="500" height="691" alt="image" src="https://github.com/user-attachments/assets/e3c5f067-df4a-46fd-906e-2d2702983cd0" />

<img width="500" height="691" alt="image" src="https://github.com/user-attachments/assets/0d11b257-08ba-4176-97db-ea0588d023b9" />

![Githubafb](https://github.com/user-attachments/assets/c8d0fe57-0e7c-460f-b1ae-3d506d95453f)


## Predictions













## Refereces 

**BibTeX format:**

```bibtex
@dataset{STARE,
  author       = {A. Hoover and V. Kouznetsova and M. Goldbaum},
  title        = {STARE: Structured Analysis of the Retina Dataset},
  year         = {2000},
  url          = {https://cecas.clemson.edu/~ahoover/stare/}
}
```

**Plain text citation:**

1. Hoover, A., Kouznetsova, V., & Goldbaum, M. (2000). *STARE: Structured Analysis of the Retina Dataset*. Retrieved from [https://cecas.clemson.edu/\~ahoover/stare/](https://cecas.clemson.edu/~ahoover/stare/)
2. Hoover, A., & Goldbaum, M. (2003). Locating the optic nerve in a retinal image using the fuzzy convergence of the blood vessels. IEEE Transactions on Medical Imaging, 22(8), 951–958.
