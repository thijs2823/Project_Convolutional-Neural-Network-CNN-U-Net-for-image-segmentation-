## Importing neccesary liberies

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import gzip
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
import sys
```

## Configuration
```python
class Config:
### Training Hyperparameters
    EPOCHS = 100
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    PREDICTION_THRESHOLD = 0.4 '''Used in the dice_score function'''
    EARLY_STOPPING_PATIENCE = 10 

### Data & Preprocessing
    TARGET_SIZE = (704, 608)
'''
Target size defines the fixed height and width to which all input images are resized,
ensuring consistency in the input dimensions for the U-Net model, which requires inputs divisible by
16 due to multiple pooling layers
'''
    CROP_DIMS = {'top': 32, 'bottom': 48, 'left': 32, 'right': 32}
    REFLECTION_THRESHOLD = 160 # Threshold to identify bright reflections

### File Paths
    # NOTE: Update these paths to your local dataset location.
    IMAGE_DIR   = r"./STARE/stare_images"
    LABEL_DIR_A = r"./STARE/stare_labels-a"
    LABEL_DIR_B = r"./STARE/stare_labels-b"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' '''Scans the device on the availability of a cuda compatible GPU'''
```

## Functions to read the images
```python
def load_image_gz(path):
    with gzip.open(path, 'rb') as f:
        img = Image.open(f).convert('L')
        img = img.resize(TARGET_SIZE[::-1], Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32)

def load_mask_gz(path):
    with gzip.open(path, 'rb') as f:
        mask = Image.open(f).convert('L') ''' Use NEAREST for masks to avoid introducing new pixel values ​​(artifacts)'''
        mask = mask.resize(TARGET_SIZE[::-1], Image.Resampling.NEAREST)
        return np.array(mask, dtype=np.float32)

def load_image_gz(path, target_size):
    """
    Loads a gzipped image as an RGB image. Returns the RGB image as a numpy array.
    """
    with gzip.open(path, 'rb') as f:
        img_pil = Image.open(f).convert('RGB') # Load as RGB
        img_pil = img_pil.resize(target_size[::-1], Image.Resampling.LANCZOS)
        img_np = np.array(img_pil, dtype=np.float32)
        return img_np

def load_mask_gz(path, target_size):
    with gzip.open(path, 'rb') as f:
        mask = Image.open(f).convert('L')
        mask = mask.resize(target_size[::-1], Image.Resampling.NEAREST)
        return np.array(mask, dtype=np.float32)

class StareMultiLabelDataset(Dataset):
    def __init__(self, image_fnames, config, is_train=False):
        self.samples = []
        self.is_train = is_train
        self.config = config

        for fname in image_fnames:
            img_path = os.path.join(self.config.IMAGE_DIR, fname)

            ''' Check for mask from the first annotator (ah) '''
            mask_fname_a = fname.replace('.ppm.gz', '.ah.ppm.gz')
            mask_path_a = os.path.join(self.config.LABEL_DIR_A, mask_fname_a)
            if os.path.exists(mask_path_a):
                self.samples.append((img_path, mask_path_a))

            ''' Check for mask from the second annotator (vk) '''
            mask_fname_b = fname.replace('.ppm.gz', '.vk.ppm.gz')
            mask_path_b = os.path.join(self.config.LABEL_DIR_B, mask_fname_b)
            if os.path.exists(mask_path_b):
                self.samples.append((img_path, mask_path_b))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        original_img = load_image_gz(
            img_path,
            self.config.TARGET_SIZE
        )
        mask = load_mask_gz(mask_path, self.config.TARGET_SIZE)

'''
Data Augmentation, for training set (this causes the model to see slightly
different versions of the same images during each epoch)
'''

        if self.is_train:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                original_img = np.fliplr(original_img).copy()
                mask = np.fliplr(mask).copy()

            # Random vertical flip
            if torch.rand(1) < 0.5:
                original_img = np.flipud(original_img).copy()
                mask = np.flipud(mask).copy()

            # Random brightness/contrast adjustment
            contrast = np.random.uniform(0.8, 1.2)
            brightness = np.random.uniform(-20, 20)
            original_img = np.clip(original_img * contrast + brightness, 0, 255)

        top_crop, bottom_crop = self.config.CROP_DIMS['top'], self.config.CROP_DIMS['bottom']
        left_crop, right_crop = self.config.CROP_DIMS['left'], self.config.CROP_DIMS['right']
        h, w, _ = original_img.shape
        if h > top_crop + bottom_crop and w > left_crop + right_crop:
            original_img = original_img[top_crop : h - bottom_crop, left_crop : w - right_crop]
            mask = mask[top_crop : h - bottom_crop, left_crop : w - right_crop]

### Normalize all channels
        original_img_norm = original_img / 255.0

### Create a 3-channel RGB tensor
        image_tensor = torch.from_numpy(original_img_norm.transpose(2, 0, 1)).float()

### Convert lists to NumPy arrays, normalize them, and make masks binar
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask[None, :, :]).float()

        return image_tensor, mask_tensor
```

## UNet setup

```python
if __name__ == '__main__':
### Loading configution parameters
    config = Config()

### Setup basic logging to the console
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
    logging.info("\n" + "="*80 + "\nSTARTING NEW TRAINING RUN\n")

### Importing image files
    all_image_fnames = sorted([f for f in os.listdir(config.IMAGE_DIR) if f.endswith(".ppm.gz")])

### Dividing data into 80% training and 20% validation
    train_fnames, val_fnames = train_test_split(all_image_fnames, test_size=0.2, random_state=42)

### Making PyTorch-dataset for train and validation
    train_dataset = StareMultiLabelDataset(train_fnames, config=config, is_train=True)
    val_dataset = StareMultiLabelDataset(val_fnames, config=config, is_train=False)

    logging.info(f" Data ready: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
```

## UNet model
```python
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
### Encoder
            self.conv1 = self.double_conv(3, 64)  # Input is 3-channel RGB
            self.conv2 = self.double_conv(64, 128)
            self.conv3 = self.double_conv(128, 256)
            self.conv4 = self.double_conv(256, 512)
            self.conv5 = self.double_conv(512, 1024)
            
### Decoder
            self.up6 = self.up_conv(1024, 512)
            self.conv6 = self.double_conv(1024, 512)
            self.up7 = self.up_conv(512, 256)
            self.conv7 = self.double_conv(512, 256)
            self.up8 = self.up_conv(256, 128)
            self.conv8 = self.double_conv(256, 128)
            self.up9 = self.up_conv(128, 64)
            self.conv9 = self.double_conv(128, 64)
            self.final = nn.Conv2d(64, 1, kernel_size=1) '''produces a 1-channel output for binary segmentation'''
            self.pool = nn.MaxPool2d(2)

        def double_conv(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_conv(self, in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 

        def forward(self, x):
            c1 = self.conv1(x)
            c2 = self.conv2(self.pool(c1))
            c3 = self.conv3(self.pool(c2))
            c4 = self.conv4(self.pool(c3))
            c5 = self.conv5(self.pool(c4))
            u6 = self.up6(c5)
            u6 = torch.cat([u6, c4], dim=1)
            c6 = self.conv6(u6)
            u7 = self.up7(c6)
            u7 = torch.cat([u7, c3], dim=1)
            c7 = self.conv7(u7)
            u8 = self.up8(c7)
            u8 = torch.cat([u8, c2], dim=1)
            c8 = self.conv8(u8)
            u9 = self.up9(c8)
            u9 = torch.cat([u9, c1], dim=1)
            c9 = self.conv9(u9)
            return self.final(c9)

    logging.info(f'Using {config.DEVICE} device')
'''
The DataLoader creates batches for training and validation, with shuffle=True ensuring randomized input order during training.
The model is moved to the appropriate device (GPU or CPU) using model.to(config.DEVICE).
'''
    num_workers = 4 if os.name == 'nt' else 8
    train_dl = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet().to(config.DEVICE)
```


## Loss functions
```python
### Focal loss
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.8, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
        def forward(self, preds, targets, weights=None):
            preds = torch.sigmoid(preds)
            bce = nn.functional.binary_cross_entropy(preds, targets, reduction='none')
            pt = torch.exp(-bce)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce

            if weights is not None:
                focal_loss = focal_loss * weights

            return focal_loss.mean()

    focal_criterion = FocalLoss()

### Tversky Loss
    class TverskyLoss(nn.Module):
        def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
            super(TverskyLoss, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.smooth = smooth
        def forward(self, preds, targets):
            preds = torch.sigmoid(preds)
            preds = preds.view(-1)
            targets = targets.view(-1)
            true_pos = (preds * targets).sum()
            false_neg = ((1 - preds) * targets).sum()
            false_pos = (preds * (1 - targets)).sum()
            tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
            return 1 - tversky_index

    tversky_criterion = TverskyLoss()
```

## Optimizer & Scheduler
```python

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
'''lowers learning rate gradually over epochs'''
```

## Dice score
```python
    def dice_score(preds, targets, eps=1e-6):
        preds = (torch.sigmoid(preds) > config.PREDICTION_THRESHOLD).float()
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)
        return dice

    def compute_loss_and_dice(prediction, labels, focal_criterion, tversky_criterion):
        """
        Calculates the combined loss and Dice score.
        """
        loss_focal = focal_criterion(prediction, labels)
        loss_tversky = tversky_criterion(prediction, labels)
        total_loss = loss_focal + loss_tversky

        dice = dice_score(prediction, labels)
        return total_loss, dice

    train_losses = []
    val_losses = []
    best_val_dice = 0.0
    early_stopping_counter = 0
    
    logging.info("Starting training")
```

## Trainings- en validatieloop
```python
    for e in tqdm(range(config.EPOCHS), leave=True, desc="Epoch"):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_dice = 0.0

### Training loop
        for data, labels in tqdm(train_dl, leave=False, desc="   Training"):
            data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            prediction = model(data)

            loss, dice = compute_loss_and_dice(prediction, labels, focal_criterion, tversky_criterion)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_dice += dice.item()

        average_train_loss = epoch_train_loss / len(train_dl)
        train_losses.append(average_train_loss)

### Validation loop
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        with torch.no_grad():
            for data, labels in tqdm(val_dl, leave=False, desc="   Validation"):
                data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
                prediction = model(data)

                loss, dice = compute_loss_and_dice(prediction, labels, focal_criterion, tversky_criterion)

                epoch_val_loss += loss.item()
                epoch_val_dice += dice.item()

        average_val_loss = epoch_val_loss / len(val_dl)
        average_val_dice = epoch_val_dice / len(val_dl)
        val_losses.append(average_val_loss)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {e+1}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, Val Dice: {average_val_dice:.4f}, LR: {current_lr:.6f}")
```

## Early stopping & logging
```python
        if average_val_dice > best_val_dice:
            best_val_dice = average_val_dice
            # A new best model is found. You could save it here.
            # For example: torch.save(model.state_dict(), "unet_best_model.pth")
            logging.info(f" New best validation Dice: {best_val_dice:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logging.info(f"No improvement in validation Dice for {early_stopping_counter} epoch(s). Patience left: {config.EARLY_STOPPING_PATIENCE - early_stopping_counter}.")
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                logging.info(f" Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} epochs without improvement.")
                break
```

## Visualize results
```python
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    logging.info("Training loss curve saved to 'training_loss_curve.png'")
```
