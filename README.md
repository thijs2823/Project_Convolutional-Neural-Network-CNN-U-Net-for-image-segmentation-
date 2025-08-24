# Project Convolutional Neural Network (CNN) / U-Net for image segmentation

My first attempt at building a Convolutional Neural Network for educational purposes, using U-Net to train on retina images and their corresponding masks.

## UNet Segmentation on STARE Dataset

This repository contains a PyTorch implementation of a UNet model for biomedical image segmentation using the STARE dataset [1,2]. The dataset consists of retina images with annotations from two different annotators. This implementation handles both annotators’ masks to improve training robustness and generalization.

## Features

* UNet architecture with BatchNorm and ReLU activations.
* Dual loss function: Focal Loss & Tversky Loss
* Data augmentation: random horizontal/vertical flips, brightness and contrast adjustments.
* Support for images with multiple annotators.
* Cropping and resizing to a consistent input size (for less noise).

## Requirements

* Python 3.10+
* PyTorch 2.x
* torchvision
* numpy
* matplotlib
* PIL
* scikit-learn
* tqdm


## Dataset Setup
The code expects the STARE dataset structured as:

```
STARE/
├── stare_images/        # .ppm.gz images
├── stare_labels-a/      # First annotator masks (.ah.ppm.gz)
└── stare_labels-b/      # Second annotator masks (.vk.ppm.gz)
```

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


## Training Progress

The following plots show the training and validation performance over 60 epochs.

<img width="500" height="691" alt="image" src="https://github.com/user-attachments/assets/e3c5f067-df4a-46fd-906e-2d2702983cd0" />

<img width="500" height="691" alt="image" src="https://github.com/user-attachments/assets/0d11b257-08ba-4176-97db-ea0588d023b9" />

![Githubafb](https://github.com/user-attachments/assets/c8d0fe57-0e7c-460f-b1ae-3d506d95453f)


## Predictions

<img width="185" height="624" alt="im0001_mask" src="https://github.com/user-attachments/assets/cbaf11d4-1561-46e7-ba5c-956f2575068a" />
<img width="245" height="605" alt="image" src="https://github.com/user-attachments/assets/30320a3d-53d0-48d9-b54c-9db7b14a00b2" />

<img width="90" height="624" alt="im0002_mask" src="https://github.com/user-attachments/assets/a11569dd-2fca-4083-a351-599070528ca5" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/d29c0f1d-9803-4685-a297-0852fa51861c" />

<img width="90" height="624" alt="im0003_mask" src="https://github.com/user-attachments/assets/9b7545b4-82d3-4e95-acfa-b13840dbaebc" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/0e0350f5-ea98-4ba5-aa7c-52b62fc513f7" />
<br>

<img width="90" height="624" alt="im0004_mask" src="https://github.com/user-attachments/assets/aa285e8e-aea8-4b8d-a541-f36257ff60c7" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/d56c43e8-e6ae-432d-aa6e-7fee28a75086" />

<img width="90" height="624" alt="im0005_mask" src="https://github.com/user-attachments/assets/47acac46-3421-4007-9f94-8fae29631662" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/9604b184-54ec-42ee-a403-6ea539db8bc8" />

<img width="90" height="624" alt="im0044_mask" src="https://github.com/user-attachments/assets/e58d96cc-5457-4749-b83b-5dedcccac9f9" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/47af456f-0476-4f44-b73b-2d253036f7b7" />

<img width="90" height="624" alt="im0077_mask" src="https://github.com/user-attachments/assets/8461b540-caf4-49ad-8393-e2d5839564da" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/1e4abf57-f78f-4162-8a87-5a1d3c4d0cf7" />
<br>

<img width="90" height="624" alt="im0081_mask" src="https://github.com/user-attachments/assets/e0f88bee-ad2b-496d-8748-208c11de6e70" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/c3a82508-bb8e-4e4d-b694-374dd83ddd5a" />

<img width="90" height="624" alt="im0082_mask" src="https://github.com/user-attachments/assets/5bdd34d3-1a69-4da3-8a7d-15fa198338c4" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/1efabd41-f9df-4bd2-a2c4-e3abcc6fd27a" />

<img width="90" height="624" alt="im0139_mask" src="https://github.com/user-attachments/assets/ba87060e-c9e4-466f-997e-e7bd922621d6" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/ec8c09e1-e988-4808-b360-374c7feb321c" />

<img width="90" height="624" alt="im0162_mask" src="https://github.com/user-attachments/assets/d2194ebf-0708-49bb-900e-a4fe7ff6af4f" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/2fb7210e-61c5-4d99-b467-48f9a953acac" />
<br>

<img width="90" height="624" alt="im0163_mask" src="https://github.com/user-attachments/assets/50a6ef3c-bbe2-40ab-8e7e-220d7fddf629" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/0aed9189-ba76-4fa1-a0ea-44c91b337b29" />

<img width="90" height="624" alt="im0235_mask" src="https://github.com/user-attachments/assets/da4961db-ceb0-4ffc-9b48-1a4317119ed9" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/3f1a91cd-9397-45be-b51e-c7cef75d5744" />

<img width="90" height="624" alt="im0236_mask" src="https://github.com/user-attachments/assets/9d9b91c4-399b-4403-a3c3-f9d572c045a6" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/c0bee3fc-9021-4bf5-823f-982419e56bea" />

<img width="90" height="624" alt="im0239_mask" src="https://github.com/user-attachments/assets/2219bc21-9988-4205-808e-68022efede90" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/be3a0bf6-7e30-41a9-b9b8-5878f7d12e1c" />
<br>

<img width="90" height="624" alt="im0240_mask" src="https://github.com/user-attachments/assets/96feb062-8423-4951-8951-06a9afac9b71" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/a51d1f42-9727-41b2-a029-fc4364db074c" />

<img width="90" height="624" alt="im0255_mask" src="https://github.com/user-attachments/assets/3b1fed52-1567-4629-bb35-da2a6144fb9a" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/6061ab80-fd18-4d79-9736-ea75e3ff50c9" />

<img width="90" height="624" alt="im0291_mask" src="https://github.com/user-attachments/assets/12dbc3ef-0794-4487-9551-2f27665393a0" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/430209a7-1230-4a37-8045-6ef5faf848db" />

<img width="90" height="624" alt="im0319_mask" src="https://github.com/user-attachments/assets/3bef3d0f-3981-4f62-b778-2fc5bc2ddc00" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/eda06186-0d49-48fe-b24b-7ddc5b49f664" />
<br>

<img width="90" height="624" alt="im0324_mask" src="https://github.com/user-attachments/assets/9cb95f0d-949a-4b7c-af60-0a56c269beff" />
<img width="120" height="605" alt="image" src="https://github.com/user-attachments/assets/498fad51-42b0-4dda-b7d0-ca98a2105b43" />


## Conclusion

The results show that some vessels identified by human annotators are missed by the model, while in other cases the model detects vessels that are not present in the manual segmentations. After 60 epochs, the model is still affected by noise (e.g. reflections). With a larger amount of training data (images and masks) and potentially higher training time, the model could be expected to perform significantly better, eventually reaching a level suitable for practical use.



## Refereces 

1. Hoover, A., Kouznetsova, V., & Goldbaum, M. (2000). *STARE: Structured Analysis of the Retina Dataset*. Retrieved from [https://cecas.clemson.edu/\~ahoover/stare/](https://cecas.clemson.edu/~ahoover/stare/)
2. Hoover, A., & Goldbaum, M. (2003). Locating the optic nerve in a retinal image using the fuzzy convergence of the blood vessels. IEEE Transactions on Medical Imaging, 22(8), 951–958.

```bibtex
@dataset{STARE,
  author       = {A. Hoover and V. Kouznetsova and M. Goldbaum},
  title        = {STARE: Structured Analysis of the Retina Dataset},
  year         = {2000},
  url          = {https://cecas.clemson.edu/~ahoover/stare/}
}
```
