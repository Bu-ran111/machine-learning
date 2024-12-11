from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
import os
from PIL import Image
import numpy as np
import glob

labal_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

labal_dict = {}

for idx,name in enumerate(labal_name):
    labal_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
])

class MyDataset(Dataset):
    def __init__(self,im_list,transform=None,loader = default_loader):
        super(MyDataset,self).__init__()
        imgs = []

        for im_item in im_list:
            #"C:\Users\ASUS\Desktop\train\airplane\aeroplane_s_000021.png"
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item,labal_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path,im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data,im_label

    def __len__(self):
        return len(self.imgs)

im_train_list = []
for category in labal_name:
    category_path = os.path.join(r"C:\Users\ASUS\Desktop\train", category, "*.png")
    im_train_list.extend(glob.glob(category_path))

im_test_list = []
for category in labal_name:
    category_path = os.path.join(r"C:\Users\ASUS\Desktop\test", category, "*.png")
    im_test_list.extend(glob.glob(category_path))

train_dataset = MyDataset(im_train_list,
                          transform = train_transform)
test_dataset = MyDataset(im_test_list,
                         transform = transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset,batch_size=6,shuffle=True,num_workers=4)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=6,shuffle=False,num_workers=4)

print("num_of_train",len(train_dataset))
print("num_of_test",len(test_dataset))