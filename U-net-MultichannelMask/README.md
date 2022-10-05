# U-Net: Semantic segmentation and classification with PyTorch
## Description
This model was trained from scratch with 12k images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 100k test images.

It can be easily used for multiclass segmentation, portrait segmentation, medical segmentation, ...


## Usage
**Note : Use Python 3.6 or newer**

### Training

```console
> python3 train.py -h

usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--load_data LOAD_DATA] [--save_data SAVE_DATA]
                [--validation VAL] [--amp] [--bilinear] [--classes CLASSES]
                [--channels CHANNELS]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --load_data LOAD_DATA
                        Load dataloaders from a .pth file
  --save_data SAVE_DATA
                        Save dataloaders into a .pth file
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --channels CHANNELS, -n CHANNELS
                        Number of input img channels

```
