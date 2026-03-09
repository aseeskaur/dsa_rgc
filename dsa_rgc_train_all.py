import numpy as np
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import torch
import random
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


filenames = Path("/home/akaur101/data/cross_validation/file_names_per_fold")
image_dir = Path(sys.argv[1])  
mask_dir = Path(sys.argv[2])
fold = 0



train_image_names = {}
test_image_names = {}

# Iterate over each text file in the folder
for file in filenames.glob('*.txt'):
    
    fold_num = int(file.name.split('_')[-1].split('.')[0])
    #print(fold_num)
    
    # Open the file
    with open(file, 'r') as f:
        # Read the lines of the file
        lines = f.readlines()
        
        filenames = [filename.strip() + 'png' for filename in lines[0].split('png') if filename.strip()]

        formatted_filenames = [f"{filename}" for filename in filenames]

        # Check if it's a train or test file based on its name
        if 'train' in file.name:
            train_image_names[fold_num] = formatted_filenames
        elif 'test' in file.name:
            test_image_names[fold_num] = formatted_filenames



 
train_image_paths = {}
test_image_paths = {}

for fold_num, filenames in train_image_names.items():
    train_image_paths[fold_num] = [image_dir / f for f in filenames]

for fold_num, filenames in test_image_names.items():
    test_image_paths[fold_num] = [image_dir / f for f in filenames]




train_mask_paths = {}
test_mask_paths = {}

for fold_num, filenames in train_image_names.items():
    train_mask_paths[fold_num] = [mask_dir / f for f in filenames]

for fold_num, filenames in test_image_names.items():
    test_mask_paths[fold_num] = [mask_dir / f for f in filenames]


train_images = train_image_paths[fold]
train_masks = train_mask_paths[fold]
test_images = test_image_paths[fold]
test_masks = test_mask_paths[fold]

train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, 
    train_masks, 
    test_size=0.2, #for validation split
    random_state=42
)


train_images_padded = []

for image in train_images:
    train_image = plt.imread(image)
    train_im_nor  = train_image/np.max(train_image)
    train_image_padded = np.pad(train_im_nor, 40, mode = 'constant')
    train_images_padded.append(train_image_padded)




train_masks_padded = []

for mask in train_masks:
    train_mask = plt.imread(mask)
    train_mas_nor = train_mask/np.max(train_mask)
    train_mask_padded  = np.pad(train_mas_nor,  40, mode = 'constant')
    train_masks_padded.append(train_mask_padded)


val_images_padded = []

for image in val_images:
    val_image = plt.imread(image)
    val_im_nor  = val_image/np.max(val_image)
    val_image_padded = np.pad(val_im_nor, 40, mode = 'constant')
    val_images_padded.append(val_image_padded)




val_masks_padded = []

for mask in val_masks:
    val_mask = plt.imread(mask)
    val_mas_nor = val_mask/np.max(val_mask)
    val_mask_padded  = np.pad(val_mas_nor,  40, mode = 'constant')
    val_masks_padded.append(val_mask_padded)




def extract_balance_patches(padded_image, padded_mask, image_name=""):
    sum_categories = {}
    x = padded_image.shape[0]
    y = padded_image.shape[1]

    for i in range(40, x-40):
        for j in range(40, y-40):
            snap_image = padded_image[i-40:i+40, j-40:j+40]
            snap_mask = padded_mask[i-1:i+2, j-1:j+2]

            mask_sum = np.sum(snap_mask)

            if mask_sum in sum_categories:
                sum_categories[mask_sum]['images'].append(snap_image)
                sum_categories[mask_sum]['masks'].append(snap_mask)
            else:
                sum_categories[mask_sum] = {
                    'images' : [snap_image],
                    'masks'  : [snap_mask]
                }
    
    print(f"\n{image_name} - Category distribution:")
    counts = []
    for category in sorted(sum_categories.keys()):
        count = len(sum_categories[category]['images'])
        print(f"  Category {category}: {count} patches")
        counts.append(count)

    n_min = min(counts)
    print(f"{image_name} - Minimum category count: {n_min}") 


    balanced_images = []
    balanced_masks  = []

    for category in sum_categories.keys():
            images_for_category = sum_categories[category]['images']
            masks_for_category = sum_categories[category]['masks']
            
            # Randomly sample n_min patches from this category
            if len(images_for_category) >= n_min:
                random_indices = random.sample(range(len(images_for_category)), n_min)
                balanced_images.extend([images_for_category[i] for i in random_indices])
                balanced_masks.extend([masks_for_category[i] for i in random_indices])

    return balanced_images, balanced_masks

def process_images(image_data, data_type=""):
    final_images = []
    final_masks = []

    for index, (padded_image, padded_mask) in enumerate(image_data):
        image_name = f"{data_type}_Image_{index+1}" if data_type else f"Image_{index+1}"
        balanced_images, balanced_masks = extract_balance_patches(padded_image, padded_mask, image_name)

        final_images.extend(balanced_images)
        final_masks.extend(balanced_masks)
    
    return final_images, final_masks



train_data = list(zip(train_images_padded, train_masks_padded))
final_train_images, final_train_masks = process_images(train_data, "Training")

val_data = list(zip(val_images_padded, val_masks_padded))
final_val_images, final_val_masks = process_images(val_data, "Validation")

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

# Replace the old dataset creation with:
train_dataset = dsa_data(image_list = final_train_images, 
                         mask_list  = final_train_masks,
                         transforms = transforms)

validation_dataset = dsa_data(image_list = final_val_images,
                              mask_list  = final_val_masks, 
                              transforms = transforms)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=50)
validation_loader = DataLoader(validation_dataset, batch_size=50)

print(f"Training dataset size: {len(final_train_images)}")
print(f"Validation dataset size: {len(final_val_images)}")
print(f"Training patches shape: {final_train_images[0].shape}")
print(f"Training masks shape: {final_train_masks[0].shape}")

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
 
print("checkpoint begin training - code running fine")   

for epoch in range(epochs):
    running_loss = 0.0
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
        if (i+1) % 100 == 0:
            print(
                 (epoch + 1, i + 1, running_loss / 100)
                 )
            running_loss_list.append(running_loss)
            running_loss = 0.0
            
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            images, masks = data
            images = images.to(device=device)
            masks = masks.to(device=device)
            val_outputs = model(images.float())
            val_loss += criterion(val_outputs, masks.float())

    # Calculate average validation loss
    avg_val_loss = val_loss / len(validation_loader)
    # Print the average validation loss for this epoch
    print(f'Epoch [{epoch + 1}/{epochs}] Validation Loss: {avg_val_loss:.4f}')

    # Check if this epoch's validation loss is the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state_dict = model.state_dict()

# Save the model with the best validation loss
if best_model_state_dict:
    torch.save(best_model_state_dict, "/home/akaur101/data/f25/dsa_rgc_bestmodel_all_dict")
    model.load_state_dict(best_model_state_dict)
    torch.save(model, "/home/akaur101/data/f25/dsa_rgc_bestmodel_all")

print('Finished Training')
