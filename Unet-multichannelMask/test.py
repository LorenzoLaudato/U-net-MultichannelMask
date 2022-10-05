import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics import Accuracy
from utils.dice_score import dice_loss, dice_coeff
from utils.utils import *
from sklearn.metrics import confusion_matrix



from utils.dataloader import HEP2Dataset
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


dir_imgs='../dataset/HEp-2_dataset/data'
csv_file='../dataset/HEp-2_dataset/final_patches.csv'
dir_save_pred='./test_outputs'



def test_net(net,
              device,
              batch_size: int = 1,
              val_percent: float = 0.1,
              load_data: int=0,
              save_data: int=0):
    if load_data==0:
        # 1. Create dataset
        dataset = HEP2Dataset(csv_file, dir_imgs)

        # 2. Split into train / validation partitions
        n_val = int(len(dataset) * val_percent)
        
        n_train = len(dataset) - n_val
        
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        n_val=int(len(val_set)*0.5)
        n_test=len(val_set)-n_val
        val_set,test_set=random_split(val_set, [n_val, n_test], generator=torch.Generator().manual_seed(0))
        # 3. Create data loaders
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)
        train_labels,val_labels,test_labels=create_labels(train_loader, val_loader, test_loader)
        if save_data==1:
            torch.save(train_loader,"data/train_loader.pth")
            torch.save(val_loader,"data/val_loader.pth")
            torch.save(test_loader,"data/test_loader.pth")
            np.save("data/train_labels.npy", train_labels)
            np.save("data/val_labels.npy", val_labels)
            np.save("data/test_labels.npy", test_labels)
            print("DATALOADERS AND LABELS SAVED!")


    elif load_data==1:
        train_loader=torch.load("data/train_loader.pth")
        val_loader=torch.load("data/val_loader.pth")
        test_loader=torch.load("data/test_loader.pth")
        train_labels=np.load("data/train_labels.npy")
        val_labels=np.load("data/val_labels.npy")
        test_labels=np.load("data/test_labels.npy")
        print("DATALOADERS AND LABELS IMPORTED!")

    
    n_train=len(train_loader)*batch_size
    n_val=len(val_loader)*batch_size
    n_test=len(test_loader)*batch_size
    print("LEN_TRAINING: ", n_train)
    print("LEN_VALIDATION: ", n_val)
    print("LEN_TEST: ", n_test)
    

    
    
    class_weights_train,class_weights_val,class_weights_test=import_class_weights(train_labels,val_labels, test_labels)
    print("TRAIN WEIGHTS: ",class_weights_train)
    print("VAL WEIGHTS: ",class_weights_val)
    print("TEST WEIGHTS: ", class_weights_test)


    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss()
    black_mask=np.zeros((384,384))
    black_mask=torch.as_tensor(black_mask, dtype=torch.float32)
    acc=Accuracy()

    # 5. Begin test
    net.eval()
    mean_seg_acc=0
    mean_cross_entropy_test=0
    dice_l=0
    y_true=[]
    y_pred=[]
    dice_score=0
    for i, item in enumerate(test_loader):
        img = item['image']/255
        true_masks=item['mask']
        label=item['label']
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred_masks,_ = net(img)
            probs = F.softmax(pred_masks, dim=1)#SOFTMAX
            one_hot=F.one_hot(probs.argmax(dim=1), 8).permute(0,3,1,2)
            output= (probs>0.45).float() #THRESHOLDING

            cross_entropy_test=criterion_val(probs.cpu(),true_masks.cpu())
            mean_cross_entropy_test+=cross_entropy_test.item()
            dice_l+=dice_loss(probs.cpu(), true_masks.cpu(), multiclass=True)
            for j in range(batch_size):
                save_image(img[j,0,:,:].float(),dir_save_pred+"/IMAGES/IMAGE_num_"+str(i*batch_size+j)+"_truelabel_"+str(label[j].item())+".png")
                max_label=find_max_1_pixel(one_hot[j,...])
                y_true.append(label[j].item())
                y_pred.append(max_label)
                compact=torch.zeros(384,384)
                for l in range(7):
                    compact+=one_hot[j,l,:,:].cpu()
                dice_score += dice_coeff(compact,true_masks[j,label[j],:,:])
                mean_seg_acc+=acc(compact.int(),true_masks[j,label[j].item(),:,:].int()).item()
                for k in range(8):   
                    save_predictions(true_masks[j,k,...],output[j,k,...],probs[j,k,...],one_hot[j,k,...],pred_masks[j,k,...],k, max_label, compact, black_mask, 37, i,j, batch_size,dir_save_pred)





    confusion_mat=confusion_matrix(y_true,y_pred)
    mca=mean_class_accuracy(confusion_mat)
    mds=dice_score/((i+1)*(j+1))
    mean_seg_acc=mean_seg_acc/((i+1)*(j+1))
    mean_cross_entropy_test=mean_cross_entropy_test/(i+1)
    print("CONFUSION MATRIX:\n",confusion_mat)
    print("MSA: ",mean_seg_acc)
    print("MCA: ",mca)
    print("MDS: ", mds.item())



def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default="MODEL.pth", help='Load model from a .pth file')
    parser.add_argument('--load_data', type=int, default=1, help='Load dataloaders from a .pth file')
    parser.add_argument('--save_data', type=int, default=0, help='Save dataloaders into a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=90.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')
    parser.add_argument('--channels', '-n', type=int, default=1, help='Number of input img channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        test_net(net=net,
                  batch_size=args.batch_size,
                  device=device,
                  val_percent=args.val/100,
                  load_data=args.load_data,
                  save_data=args.save_data)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
    