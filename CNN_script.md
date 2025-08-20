## Importing neccesary liberies

```python
import os
import gzip
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
```

#### The TARGET_SIZE defines the fixed height and width to which all input images are resized, ensuring consistency in the input dimensions for the U-Net model, which requires inputs divisible by 16 due to multiple pooling layers
```python
TARGET_SIZE = (704, 608) # (height, width)
```

## Functions to read the images
```python
def load_image_gz(path):
    with gzip.open(path, 'rb') as f:
        img = Image.open(f).convert('L') # PIL.resize heeft (width, height) nodig, dus we draaien de tuple om.
        img = img.resize(TARGET_SIZE[::-1], Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32)

def load_mask_gz(path):
    with gzip.open(path, 'rb') as f:
        mask = Image.open(f).convert('L')
        # Gebruik NEAREST voor maskers om te voorkomen dat er nieuwe pixelwaarden (artefacten) ontstaan.
        mask = mask.resize(TARGET_SIZE[::-1], Image.Resampling.NEAREST)
        return np.array(mask, dtype=np.float32)
```

## Loading data
```python
images, masks = [], []
for fname in os.listdir(image_dir):
    if fname.endswith(".ppm.gz"):
        img_path = os.path.join(image_dir, fname)
        mask_fname = fname.replace('.ppm.gz', '.ah.ppm.gz')  # voor 'a' labels
        mask_path = os.path.join(label_dir, mask_fname)
        if os.path.exists(mask_path):
            images.append(load_image_gz(img_path))
            masks.append(load_mask_gz(mask_path))

if not images:
    print("Fout: Geen overeenkomende image/mask-paren gevonden. Controleer de bestandspaden en -namen.")
    exit()

print("Images geladen:", len(images))
print("Masks geladen:", len(masks))
```


## Convert lists to NumPy arrays, normalize them, and make masks binary.
```python
images = np.array(images, dtype=np.float32) / 255.0
masks = np.array(masks, dtype=np.float32) / 255.0
masks = (masks > 0.5).astype(np.float32)
```

#### Voeg channel dimension toe voor PyTorch [N, C, H, W]

```python
if images.ndim == 3:
    images = images[:, None, :, :]
if masks.ndim == 3:
    masks = masks[:, None, :, :]

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
```

## Train/Val split
```python
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32) ``` Naar torch tensors ```
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
```

## U-Net model
```python
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = self.double_conv(1, 64)   # 1 kanaal (Grayscale), niet 3
        self.conv2 = self.double_conv(64, 128)
        self.conv3 = self.double_conv(128, 256)
        self.conv4 = self.double_conv(256, 512)
        self.conv5 = self.double_conv(512, 1024)

        self.up6 = self.up_conv(1024, 512)
        self.conv6 = self.double_conv(1024, 512)
        self.up7 = self.up_conv(512, 256)
        self.conv7 = self.double_conv(512, 256)
        self.up8 = self.up_conv(256, 128)
        self.conv8 = self.double_conv(256, 128)
        self.up9 = self.up_conv(128, 64)
        self.conv9 = self.double_conv(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(F.max_pool2d(c1, 2))
        c3 = self.conv3(F.max_pool2d(c2, 2))
        c4 = self.conv4(F.max_pool2d(c3, 2))
        c5 = self.conv5(F.max_pool2d(c4, 2))

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

        return self.final(c9) # Sigmoid wordt nu afgehandeld door BCEWithLogitsLoss
```

## Training + Validation
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss() # Numeriek stabieler dan Sigmoid + BCELoss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 18
best_val_loss = float("inf")
train_losses, val_losses = [], [] 
```

### Training
```python
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
``` 

### Validation
```python
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
```

### Store losses
```python
    train_losses.append(train_loss)
    val_losses.append(val_loss)
```

### Print resultaten
```python
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
```

### Saving best model
```python
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "unet_best_stare.pth")
        print(">> Model opgeslagen (nieuw beste val loss)")
```
