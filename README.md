# Age Estimation via Fine-tuned ResNet-50

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kr1347/ResNet-Model/blob/main/ResNet_model.ipynb)

Facial age estimation using a fine-tuned ResNet-50 regression model trained on the UTKFace dataset. Predicts continuous age values from face images.

## Architecture

ResNet-50 (pretrained on ImageNet) with the final fully connected layer replaced by a regression head:

```
FC(2048 → 256) → ReLU → Dropout(0.3) → FC(256 → 1)
```

## Dataset

**UTKFace** — Large-scale face dataset with age labels spanning 0–116 years. Images are labeled with age, gender, and ethnicity encoded in filenames.

- Train/Test split: 80/20 (stratified random)
- Input resolution: 224×224 (ResNet standard)

## Training

| Hyperparameter | Value |
|---------------|-------|
| Loss | MSELoss |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 30 |
| Batch size | 32 |
| Augmentation | RandomCrop, ColorJitter, HorizontalFlip |

ImageNet normalization applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

## Results

Training loss monitored across 30 epochs with validation set evaluation per epoch. Transfer learning from ImageNet features significantly accelerates convergence compared to training from scratch.

## Usage

Open the notebook in Colab. Mount Google Drive and provide the UTKFace dataset zip path. Run all cells sequentially.

## References

- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*
- Zhang, Z. et al. (2017). Age progression/regression by conditional adversarial autoencoder. *CVPR 2017* (UTKFace dataset)

