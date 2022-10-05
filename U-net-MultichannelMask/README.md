# U-Net: Semantic segmentation and classification with PyTorch

## Quick start

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```
pip3 install -r requirements.txt
```

4. Run training:
```
python3 train.py
```
5. Run testing and visualize predictions:
```
python3 test.py
```

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
### Testing
```console
> python3 test.py -h
usage: test.py [-h] [--batch-size B] [--load LOAD] [--load_data LOAD_DATA]
               [--save_data SAVE_DATA] [--validation VAL] [--bilinear]
               [--classes CLASSES] [--channels CHANNELS]

Test the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --batch-size B, -b B  Batch size
  --load LOAD, -f LOAD  Load model from a .pth file
  --load_data LOAD_DATA
                        Load dataloaders from a .pth file
  --save_data SAVE_DATA
                        Save dataloaders into a .pth file
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --channels CHANNELS, -n CHANNELS
                        Number of input img channels

```

