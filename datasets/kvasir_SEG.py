import os
import os.path as osp
from utils.transform import *
from torch.utils.data import Dataset
from torchvision import transforms


# KavSir-SEG Dataset
class kvasir_SEG(Dataset):
    def __init__(self, root, data2_dir, mode='train', transform=None):
        super(kvasir_SEG, self).__init__()
        data_path = osp.join(root, data2_dir)
        self.imglist = []
        self.gtlist = []
        self.mode=mode
        self.totesor=transforms.ToTensor()
        self.resize=transforms.Resize((320,320))
        datalist = os.listdir(osp.join(data_path, 'images'))
        for data in datalist:
            self.imglist.append(osp.join(data_path+'/images', data))
            self.gtlist.append(osp.join(data_path+'/masks', data))

        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((320, 320)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   #Translation(10),
                   RandomCrop((224, 224)),
                   ToTensor(),

               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((320, 320)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        # print(gt_path)
        # print(gt_path.split('masks/')[1])
        name=gt_path.split('masks/')[1]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        #print(data)
        if self.transform and self.mode=='train':
            data = {'image': img, 'label': gt}
            data = self.transform(data)
        elif self.mode == 'valid' or self.mode == 'test':
            # img=transforms.Resize((320,320))
            # img=transforms.ToTensor()
            # gt=transforms.ToTensor()
            img=self.resize(img)
            img=self.totesor(img)
            gt=self.totesor(gt)
            #gt=self.resize(gt)
            data = {'image': img, 'label': gt}
            #data = self.transform(data)
            #print(type(data))
            data['name']=name
            #data.add(f'name:{name}')
            # print(data)
            # get()
            #data.add('name':name)
        return data

    def __len__(self):
        return len(self.imglist)