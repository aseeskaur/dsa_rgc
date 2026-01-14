# Edited this on Sep 30
# Added: plots for training and validation losses and related lists for that
# changed the path to save the trained models and their dict

import numpy as np
import pandas as pd
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torch
import random

from pathlib import Path
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

images_path = sys.argv[1]
masks_path  = sys.argv[2]

train_image_path  = Path(images_path)/"01.png"
train_mask_path   = Path(masks_path)/"01.png"

val_image_path    = Path(images_path)/"16.png"
val_mask_path     = Path(masks_path)/"16.png"

test_image_path   = Path(images_path)/"10.png"
test_mask_path    = Path(masks_path)/"10.png"

train_image = plt.imread(train_image_path)
train_mask  = plt.imread(train_mask_path)

val_image   = plt.imread(val_image_path)
val_mask    = plt.imread(val_mask_path)

test_image  = plt.imread(test_image_path)
test_mask   = plt.imread(test_mask_path)


train_im_nor  = train_image/np.max(train_image)
train_mas_nor = train_mask/np.max(train_mask)

val_im_nor    = val_image/np.max(val_image)
val_mas_nor   = val_mask/np.max(val_mask)

test_im_nor   = test_image/np.max(test_image)
test_mas_nor  = test_mask/np.max(test_mask)

train_image_padded = np.pad(train_im_nor, 40, mode = 'constant')
train_mask_padded  = np.pad(train_mas_nor,  40, mode = 'constant')

val_image_padded   = np.pad(val_im_nor, 40, mode = 'constant')
val_mask_padded    = np.pad(val_mas_nor,  40, mode = 'constant')

test_image_padded = np.pad(test_im_nor, 40, mode = 'constant')
test_mask_padded  = np.pad(test_mas_nor,  40, mode = 'constant')

#**Making 80 X 80 snaps for training images and 3 X 3 snaps for training masks**

sum_categories_tr = {}
sum_categories_val = {}

train_image_list = []
train_mask_list  = []

val_image_list = []
val_mask_list  = []

snap_image_tr = np.empty([80, 80])
snap_mask_tr  = np.empty([3, 3])

snap_image_val = np.empty([80, 80])
snap_mask_val  = np.empty([3, 3])

x =  np.shape(train_image_padded)[0]
y =  np.shape(train_image_padded)[1]

for i in range(40,x-40):
    for j in range(40,y-40):
        current_tr    = train_image_padded[i][j]
        current_val    = val_image_padded[i][j]
        
        snap_image_tr = train_image_padded[i-40:i+40, j-40:j+40]
        snap_image_val = val_image_padded[i-40:i+40, j-40:j+40]
        
        snap_mask_tr  = train_mask_padded[i-1:i+2, j-1:j+2]
        snap_mask_val  = val_mask_padded[i-1:i+2, j-1:j+2]
        
        train_image_list.append(snap_image_tr)
        val_image_list.append(snap_image_val)
        
        train_mask_list.append(snap_mask_tr)
        val_mask_list.append(snap_mask_val)
        
        snap_mask_tr_sum = np.sum(snap_mask_tr)
        snap_mask_val_sum = np.sum(snap_mask_val)
        
        
        
        if snap_mask_tr_sum in sum_categories_tr:
            sum_categories_tr[snap_mask_tr_sum]['images'].append(snap_image_tr)
            sum_categories_tr[snap_mask_tr_sum]['masks'].append(snap_mask_tr)
        else:
            sum_categories_tr[snap_mask_tr_sum] = {'images': [snap_image_tr], 'masks': [snap_mask_tr]}
            
        if snap_mask_val_sum in sum_categories_val:
            sum_categories_val[snap_mask_val_sum]['images'].append(snap_image_val)
            sum_categories_val[snap_mask_val_sum]['masks'].append(snap_mask_val)
        else:
            sum_categories_val[snap_mask_val_sum] = {'images': [snap_image_val], 'masks': [snap_mask_val]}
            
n = 2000

random_tr_images = []
random_tr_masks  = []

random_val_images = []
random_val_masks  = []

for sum_value_tr in sum_categories_tr:
    tr_images_for_category = sum_categories_tr[sum_value_tr]['images']
    tr_masks_for_category = sum_categories_tr[sum_value_tr]['masks']

    # Randomly pick 2000 images and their corresponding masks
    if len(tr_images_for_category) >= n:
        random_tr_ind = random.sample(range(len(tr_images_for_category)), n)
        random_tr_images.extend([tr_images_for_category[i] for i in random_tr_ind])
        random_tr_masks.extend([tr_masks_for_category[i] for i in random_tr_ind])


for sum_value_val in sum_categories_val:
    val_images_for_category = sum_categories_val[sum_value_val]['images']
    val_masks_for_category = sum_categories_val[sum_value_val]['masks']

    # Randomly pick 2000 images and their corresponding masks
    if len(val_images_for_category) >= n:
        random_val_ind = random.sample(range(len(val_images_for_category)), n)
        random_val_images.extend([val_images_for_category[i] for i in random_val_ind])
        random_val_masks.extend([val_masks_for_category[i] for i in random_val_ind])
        
        
class dsa_data(Dataset):
    def __init__(self, image_list, mask_list, transforms):
        self.image_list = image_list
        self.mask_list  = mask_list
        self.transforms = transforms
        
    def __len__(self):
        
        return len((self.image_list))
    
    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask  = self.mask_list[index].astype(float)
        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
#         image = image[None, :]
#         mask  = mask[None, :]
        
        return (image, mask)

#**Defining the transforms**

transforms = transforms.ToTensor()

#**Defining the training dataset and training dataloader**

train_dataset = dsa_data(image_list = random_tr_images, 
                         mask_list  = random_tr_masks,
                         transforms = transforms)
train_loader  = DataLoader(train_dataset, shuffle = True, batch_size = 50)

# Define validation dataset and dataloader

validation_dataset = dsa_data(image_list = random_val_images,
                              mask_list  = random_val_masks, 
                              transforms = transforms)
validation_loader = DataLoader(validation_dataset, batch_size = 50)


#**Defining the testing dataset and testing dataloader**



print("checkpoint dataloaders - code running fine")

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
model  = net()
model  = model.float()
model  = model.to(device=device)

optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

#**Training the model**

running_loss_list = []


epochs = 50



# Initialize variables to keep track of the best validation loss
best_val_loss = float('inf')
best_model_state_dict = None
training_losses =[]
validation_losses = []

print("checkpoint begin training - code running fine")   

for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    model.train()  # Set the model to training mode
    for i, data in enumerate(train_loader):
        images, masks = data
        images = images.to(device=device)
        masks = masks.to(device=device)
        optimizer.zero_grad()    #empty the gradients
        
        outputs = model(images.float())
        #print(outputs)
        loss = criterion(outputs, masks.float())
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()

        if (i+1) % 100 == 0:
            print(
                 (epoch + 1, i + 1, running_loss / 100)
                 )
            running_loss_list.append(running_loss)
            running_loss = 0.0
        
    avg_train_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            images, masks = data
            images = images.to(device=device)
            masks = masks.to(device=device)
            val_outputs = model(images.float())
            val_loss += criterion(val_outputs, masks.float()).item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(validation_loader)
    validation_losses.append(avg_val_loss)

    # Print the average validation loss for this epoch
    print(f'Epoch [{epoch + 1}/{epochs}] Validation Loss: {avg_val_loss:.4f}')

    # Check if this epoch's validation loss is the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state_dict = model.state_dict()

# Save the model with the best validation loss
if best_model_state_dict:
    torch.save(best_model_state_dict, "/home/akaur101/data/f25/trained_dsa_models/dsa_rgc_single_image_bestmodel_dict")
    model.load_state_dict(best_model_state_dict)
    torch.save(model, "/home/akaur101/data/f25/trained_dsa_models/dsa_rgc_single_image_bestmodel")

print('Finished Training')
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss - Trained on Single Image')
plt.legend()
plt.savefig(f'/home/akaur101/data/f25/dsa_results/loss_plot_single_image.png')
plt.show()
