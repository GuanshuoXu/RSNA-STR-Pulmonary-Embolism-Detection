# RSNA-STR-Pulmonary-Embolism-Detection

Link to method description:
https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/194145


Software Requirement:
torch 1.3.1
torchvision 0.4.2
pydicom 2.0.0
pandas 1.0.3
numpy 1.17.2
gdcm 2.8.9
apex 0.1
huggingface transformers 2.11.0
imgaug

Following libraries are used in my code but not required to install:
https://github.com/Cadene/pretrained-models.pytorch
https://github.com/lukemelas/EfficientNet-PyTorch
https://github.com/albumentations-team/albumentations

Hardware Requirement:
I used 4 x RTX Titan for image-level model training, and 1 x 2070 for study-level training.

For local validation, go to trainval/ and sh run.sh

To retrain on the full training data, go to trainall/ and sh run.sh
