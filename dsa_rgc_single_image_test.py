
import numpy as np
import pandas as pd
import torch.nn as nn
import sys
import matplotlib.pyplot as plt

import torch
import random
import os

from pathlib import Path
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

images_path = sys.argv[1]
masks_path  = sys.argv[2]

test_image_path   = Path(images_path)/"10.png"
test_mask_path    = Path(masks_path)/"10.png"

test_image  = plt.imread(test_image_path)
test_mask   = plt.imread(test_mask_path)

test_im_nor   = test_image/np.max(test_image)
test_mas_nor  = test_mask/np.max(test_mask)

test_image_padded = np.pad(test_im_nor, 40, mode = 'constant')
test_mask_padded  = np.pad(test_mas_nor,  40, mode = 'constant')

#**Defining the testing dataset and testing dataloader**


#**CNN Model**

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(num_features = out_channels),
                    nn.ReLU(inplace = True),
                    nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(num_features = out_channels),
                    nn.ReLU(inplace = True))
        
    def forward(self, x):
        return self.conv(x)
    

class net(nn.Module):
    
    def __init__(self):
        super(net, self).__init__()
        
        
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride= 3)
        self.conv1    = double_conv(1, 64)
        self.conv2    = double_conv(64,128)
        self.conv3    = double_conv(128,256)
        self.conv4    = double_conv(256, 512)
        self.conv5    = double_conv(512,1024)
        self.conv6    = nn.Conv2d(1024, 1, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, image):
        
        x1  = self.conv1(image)
        #print("x1: ", x1.size())
        x2  = self.maxpool(x1)
        #print(x2.size())
        x3  = self.conv2(x2)
        #print(x3.size())
        x4  = self.maxpool(x3)
        #print(x4.size())
        x5  = self.conv3(x4)
        #print(x5.size())
        x6  = self.maxpool(x5)
        #print(x6.size())
        x7  = self.conv4(x6)
        #print(x7.size())
        x8  = self.maxpool2(x7)
        #print(x8.size())
        x9  = self.conv5(x8)
        #print(x9.size())
        x10 = self.conv6(x9)
        #print(x10.size())
        return(x10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
#device = torch.device('cpu')
model  = net()
model  = model.float()



model  = model.to(device=device)
model.load_state_dict(torch.load("/home/akaur101/data/f25/dsa_rgc_single_image_bestmodel_dict", map_location=device))
model.eval()

##Testing


    

num = 500
centroid_df = pd.DataFrame()

rand_ele      = np.random.choice(test_image.flatten(), num, replace = False)
indices       = np.argwhere(np.isin(test_image, rand_ele)).tolist()
centroids_all = random.choices(indices, k = num)
centroid_df   = pd.DataFrame(centroids_all[0:], columns = ['row', 'col'])
centroid_df['row']= centroid_df['row']+40
centroid_df['col']= centroid_df['col']+40

#give it a list of all next centroids
def img_process(next_cents, padded_image, padded_mask):
    image_list = []
    mask_list  = []
    val_list   = []
    rows       = []
    cols       = []

    snap_image = np.empty([80, 80])
    snap_mask  = np.empty([3, 3])
    
    for i in next_cents:
        row = i[0]
        col = i[1]
        val = padded_image[row][col]
        snap_image = padded_image[row-40:row+40, col-40:col+40]
        snap_mask  = padded_mask[row-1:row+2, col-1:col+2]
        image_list.append(snap_image)
        mask_list.append(snap_mask)
        val_list.append(val)
        rows.append(row)
        cols.append(col)
    
    return image_list, mask_list, val_list, rows, cols

class newcelldata(Dataset):
    def __init__(self, image_list, mask_list, rows, cols, transforms):
        self.image_list = image_list
        self.mask_list  = mask_list
        self.rows       = rows
        self.cols       = cols
        self.transforms = transforms
        
    def __len__(self):
        
        return len((self.image_list))
    
    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask  = self.mask_list[index].astype(float)
        rows  = self.rows[index]
        cols  = self.cols[index]
        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask, rows, cols)

transforms = transforms.ToTensor()

S = []   #master centroid list
runn_cent = []
next_S = []
df = centroid_df
#df = new_cent
all_pred = []
all_indices = []
row_list = []
col_list = []

for i in df.index:
    row = df['row'][i]
    col = df['col'][i]
    next_S.append([row, col])
#S_0 is the first next_S

S.append(next_S)
counter = 0
while len(next_S) > 0:
    counter+=1
    print(len(next_S))
    test_image_list, test_mask_list, val_list, rows, cols = img_process(next_S,test_image_padded, test_mask_padded)
    test_dataset = newcelldata(image_list = test_image_list, 
                               mask_list  = test_mask_list,
                               rows = rows,
                               cols = cols,
                               transforms = transforms)
    test_loader  = DataLoader(test_dataset, 50, shuffle=False)
    
    with torch.no_grad():
        #import pdb; pdb.set_trace()
        for k, data in enumerate(test_loader):
            #print(k)
            images, masks, rows, cols = data
            images = images.to(device = device)
            masks = masks.to(device = device)
            pred = model(images.float())
            final_pred = torch.sigmoid(pred)
            #print(final_pred)
            
            pred_temp = torch.where(torch.squeeze(final_pred) < 0.1, 0, 1)
            if len(final_pred) == 1:
                pred_temp = torch.where((final_pred) < 0.1, 0, 1)
            
            pred_ind = torch.argwhere(pred_temp)
            #import pdb; pdb.set_trace()
            #all_indices.append(pred_ind)
            #all_pred.append(pred_temp[pred_ind])
            temp_ind_list = []
            temp_val_list = []
            temp_current = []
            for i in range(len(pred_ind)): 
                current = pred_ind[i][0].item()
                
                row_item = rows[current].item()
                pred_ind_row = pred_ind[i][1].item()
                x = row_item - 1 + pred_ind_row
    
                col_item = cols[current].item()
                pred_ind_col = pred_ind[i][2].item()
                y = col_item - 1 + pred_ind_col
                
                boundary_size = 40
    
                # Check if the indices are within the image boundaries, excluding the 40-pixel boundary
                if boundary_size <= x < test_image_padded.shape[0] - boundary_size and \
                   boundary_size <= y < test_image_padded.shape[1] - boundary_size:
                    val = torch.squeeze(final_pred[current])[pred_ind_row, pred_ind_col]
                    temp_current.append(current)
                    temp_ind_list.append([x, y])
                    temp_val_list.append(val)

                # If the indices are out of bounds, you can skip or handle them as needed
                else:
                    continue
            all_indices.append(temp_ind_list)
            all_pred.append(temp_val_list)
        
            for i in temp_ind_list:
                if i not in runn_cent:
                    runn_cent.append(i)    
            
            #print(runn_cent)
            
        #print(runn_cent)
    next_S = []
    for i in runn_cent:
        if i not in S:
            #print(i)
            next_S.append(i)
    S.extend(next_S)
print("done testing")

save_dir = "/home/akaur101/data/f25"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "all_pred_list"), 'w') as fp:
    for item in all_pred:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done with all_pred')
    
with open(os.path.join(save_dir, "all_ind_list"), 'w') as fp:
    for item in all_indices:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done with all_indices')
recon_mask = np.zeros_like(test_mask_padded)
flat_list_ind = [item for sublist in all_indices for item in sublist]
flat_list_pred = [item for sublist in all_pred for item in sublist]
for i in range(len(flat_list_ind)):
    x = flat_list_ind[i][0]
    y = flat_list_ind[i][1]
    #recon_mask[x, y] = flat_list_pred[i]
    recon_mask[x, y] = 1

new_recon_mask = recon_mask[40:-40, 40:-40]
plt.figure()
plt.imshow(new_recon_mask, cmap='gray')
plt.savefig(os.path.join(save_dir, "new_recon_mask.png"))
plt.close()
print("saved cropped recon mask")


