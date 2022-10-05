# U-Net: Semantic segmentation with PyTorch

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

## Usage
**Note : Use Python 3.6 or newer**

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

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Test

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

`python3 test.py `
In this way you can run the model on a test set of 1,5K images, save the output masks in test_outputs/ and show some metrics.

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
---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
