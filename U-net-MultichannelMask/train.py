import argparse
import logging
from sklearn.metrics import confusion_matrix
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from torchmetrics import Accuracy
from sklearn.utils import class_weight
from utils.utils import *


from utils.dataloader import HEP2Dataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


dir_checkpoint = Path('./checkpoints/')
dir_save_pred='./outputs'
dir_save_pred_1='./outputs/IMAGES'
dir_save_pred_2='./outputs/TRUE_MASKS'
dir_save_pred_3='./outputs/PRED_MASKS'
dir_imgs='../dataset/HEp-2_dataset/data'
csv_file='../dataset/HEp-2_dataset/final_patches.csv'

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False,
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
            torch.save(train_loader,"train_loader8.pth")
            torch.save(val_loader,"val_loader8.pth")
            torch.save(test_loader,"test_loader8.pth")
            np.save("train_labels8.npy", train_labels)
            np.save("test_labels8.npy", test_labels)
            np.save("val_labels8.npy", val_labels)
            print("DATALOADERS AND LABELS SAVED!")

    elif load_data==1:
        train_loader=torch.load("train_loader8.pth")
        val_loader=torch.load("val_loader8.pth")
        test_loader=torch.load("test_loader8.pth")
        train_labels=np.load("train_labels8.npy")
        val_labels=np.load("val_labels8.npy")
        test_labels=np.load("test_labels8.npy")
        print("DATALOADERS AND LABELS IMPORTED!")
    #print("ADDED 8 CLASSES. ADDED DICE_LOSS. NO WEIGHTS ON CLASSES IN CROSSENTROPY. LR= 1e-04 WD=1e-05 MOMENTUM=0.9 PATIENCE=3")
    n_train=len(train_loader)*batch_size
    n_val=len(val_loader)*batch_size
    n_test=len(test_loader)*batch_size
    print("LEN_TRAINING: ", n_train)
    print("LEN_VALIDATION: ", n_val)
    print("LEN_TEST: ", n_test)
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-5, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)  # goal: minimize loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    black_mask=np.zeros((384,384))
    black_mask=torch.as_tensor(black_mask, dtype=torch.float32)
    
    
    class_weights_train,class_weights_val,class_weights_test=import_class_weights(train_labels,val_labels, test_labels)

    print("TRAIN WEIGHTS: ",class_weights_train)
    print("VAL WEIGHTS: ",class_weights_val)
    print("TEST WEIGHTS: ", class_weights_test)

    #define CrossEntropyLosses e Accuracy
    criterion_train = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    criterion_test = nn.CrossEntropyLoss()
    acc=Accuracy()

    global_step = 0
    writer=SummaryWriter()
    if dir_save_pred is not None and not os.path.exists(dir_save_pred):
        Path(dir_save_pred).mkdir(parents=True, exist_ok=True)
        Path(dir_save_pred /Path("/IMAGES/")).mkdir(parents=True, exist_ok=True)
        Path(dir_save_pred / Path("/TRUE_MASKS/")).mkdir(parents=True, exist_ok=True)
        Path(dir_save_pred / Path("/PRED_MASKS/")).mkdir(parents=True, exist_ok=True)


    # 5. Begin training
    for epoch in range(1, epochs+1):
        if epoch%1==0:
            net.eval()
            mean_acc=0
            mean_loss=0
            y_true=[]
            y_pred=[]
            for i, item in enumerate(val_loader):
                img = item['image']/255
                true_masks=item['mask']
                label=item['label']
                img = img.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    pred_masks,_ = net(img)
                    probs = F.softmax(pred_masks, dim=1)#SOFTMAX
                    one_hot=F.one_hot(probs.argmax(dim=1), net.n_classes).permute(0,3,1,2)
                    output= (probs>0.45).float() #THRESHOLDING
                    mean_seg_acc+=acc(one_hot.cpu().int(),true_masks.cpu().int()).item()
                    loss_val=criterion_val(probs.cpu(),true_masks.cpu())+dice_loss(probs.cpu(), true_masks.cpu(), multiclass=True)
                    mean_loss+=loss_val.item()
                    if i>=4:
                        break
                    for j in range(batch_size):
                        save_image(img[j,0,:,:].float(),dir_save_pred+"/IMAGES/IMAGE_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_truelabel_"+str(label[j].item())+".png")
                        max_label=find_max_1_pixel(one_hot[j,...])
                        y_true.append(label[j])
                        y_pred.append(max_label)
                        # if max_label==label[j]:
                        #     accuracies=update_accuracies(max_label,label[j], accuracies,all=True)
                        # else:
                        #     accuracies=update_accuracies(max_label, label[j], accuracies)
                        compact=torch.zeros(384,384)
                        for l in range(7):
                            compact+=one_hot[j,l,:,:].cpu()
                        for k in range(8):   
                            save_predictions(true_masks[j,k,...],output[j,k,...],probs[j,k,...],one_hot[j,k,...],pred_masks[j,k,...],k, max_label, compat, black_mask, epoch, i,j, batch_size,dir_save_pred)


            confusion_mat= confusion_matrix(y_true,y_pred)
            mean_seg_acc=mean_seg_acc/(i+1)
            mean_loss=mean_loss/(i+1)
            mca=mean_class_accuracy(confusion_mat)
            print("MSA: ",mean_seg_acc)
            print("MCA: ", mca)
            print("MEAN_LOSS: ",mean_loss)
            print("CONFUSION MATRIX: ", confusion_mat)

            writer.add_scalar("MSA/val",mean_acc,epoch)
            writer.add_scalar("MEAN_LOSS/val",mean_loss,epoch)
            #writer.add_scalar("MCA/val", mca, epoch)
    
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i,item in enumerate(train_loader):
                image= item['image']/255
                true_masks = item['mask']
                label=item['label']
                label=label.to(device=device,dtype=torch.int16)

                image = image.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                assert image.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                with torch.cuda.amp.autocast(enabled=amp):
                    pred_masks, _= net(image) 
                    probs=F.softmax(pred_masks, dim=1)
                    loss=criterion_train(probs.cpu(),true_masks.cpu())+ dice_loss(probs.cpu(), true_masks.cpu(), multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(image.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        val_loss_mean = evaluate(batch_size,net, val_loader, device, criterion_val)
                        scheduler.step(val_loss_mean)
                        logging.info('Validation Cross Entropy: {}'.format(val_loss_mean))
       
        writer.add_scalar("EPOCH_LOSS/train",epoch_loss/len(train_loader),epoch)
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}_{}.pth'.format(epoch,epoch_loss/len(train_loader))))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--load_data', type=int, default=1, help='Load dataloaders from a .pth file')
    parser.add_argument('--save_data', type=int, default=0, help='Save dataloaders into a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
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
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val/100,
                  load_data=args.load_data,
                  save_data=args.save_data)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
    