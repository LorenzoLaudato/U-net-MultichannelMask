from calendar import c
from distutils.command.config import config
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils import class_weight
from torchvision.utils import save_image
from pathlib import Path
import os



def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def find_max_1_pixel(pred_masks):
    max_label=0
    max_pixel=0
    for i in range(1, 7):
        pixel_found=(147456-np.unique(pred_masks[i,...].cpu(), return_counts=True)[1][0])
        if pixel_found>max_pixel:
            max_pixel=pixel_found
            max_label=i
    return max_label

def find_max_mask(pred_masks):
    max_label=0
    max_sum=0.0
    for i in range(7):
        sum=torch.sum(pred_masks[i,...]).item()
        if sum>max_sum:
            max_sum=sum
            max_label=i
    return max_label
    
def update_accuracies(pred_label,true_label, accuracies, all=False):
    for i in range(7):
        if not all:
            if i not in [pred_label, true_label]:
                accuracies[i]+=1
        else:
            accuracies[i]+=1
    return accuracies


def mean_class_accuracy(confusion_matrix):
    count=confusion_matrix.sum(axis=1)
    true=np.diag(confusion_matrix)
    return np.mean(true/count)

    
def import_class_weights(train_labels,val_labels,test_labels):
    classes,_=np.unique(train_labels, return_counts=True)
    class_weights_train=class_weight.compute_class_weight('balanced',classes,train_labels)
    class_weights_train=torch.tensor(class_weights_train,dtype=torch.float32)
    class_weights_val=class_weight.compute_class_weight('balanced',classes,val_labels)
    class_weights_val=torch.tensor(class_weights_val,dtype=torch.float32)
    class_weights_test=class_weight.compute_class_weight('balanced',classes,test_labels)
    class_weights_test=torch.tensor(class_weights_test,dtype=torch.float32)
    return class_weights_train, class_weights_val, class_weights_test
def create_labels(train_loader, val_loader, test_loader, batch_size):
    train_labels=[]
    for item in train_loader:
        label = item['label']
        for i in range(batch_size):
            train_labels.append(label[i])
    train_labels=np.array(train_labels)
    val_labels=[]
    for item in val_loader:
        label = item['label']
        for i in range(batch_size):
            val_labels.append(label[i])
    val_labels=np.array(val_labels)
    test_labels=[]
    for item in test_loader:
        label = item['label']
        for i in range(batch_size):
            test_labels.append(label[i])
    test_labels=np.array(test_labels)
    return train_labels,val_labels,test_labels

def save_predictions(true_mask,output,probs,one_hot,pred_mask,k, max_label, compat, black_mask, epoch, i,j, batch_size, dir_save_pred):
    save_image(true_mask,dir_save_pred + "/TRUE_MASKS/TRUEMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    save_image(output.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK(OUTPUT)_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    save_image(probs.float(),dir_save_pred + "/PRED_MASKS/SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    save_image(one_hot.float(),dir_save_pred + "/PRED_MASKS/ONEHOT_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    save_image(pred_mask.float(),dir_save_pred + "/PRED_MASKS/OUTNET_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    if k==max_label:
        save_image(compat.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    elif k==7:
        save_image(one_hot.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    else:
        save_image(black_mask,dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")