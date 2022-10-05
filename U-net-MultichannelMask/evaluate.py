import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn


from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss

def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice 

def evaluate(batch_size,net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    loss=0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches*batch_size, desc='Validation round', unit='batch', leave=False):
        image, mask_true,label = batch['image']/255, batch['mask'], batch['label']
        label=label.to(device=device,dtype=torch.int16)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        output, _ = net(image)
        probs= F.softmax(output, dim=1)#SOFTMAX
        with torch.no_grad():
            mean_loss=criterion(probs.cpu(),mask_true.cpu())+dice_loss(probs.cpu(),mask_true.cpu(), multiclass=True)
            loss += mean_loss.item()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return loss
    return loss/ num_val_batches
