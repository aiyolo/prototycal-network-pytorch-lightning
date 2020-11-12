from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class MiniImageNet(Dataset):
    def __init__(self, setname='train'):
        self.data_path, self.label = self.load_data_info(setname)  # 38400
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        img_path = self.data_path[i]
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.label[i] 
        return img, label

    def load_data_info(self, setname):
        # data_root = '/media/data/MiniImageNet/'
        data_root = '/home/aiyolo/PycharmProjects/LearningToCompare_FSL/datas/miniImagenet'
        if setname=='train':
            THE_PATH = os.path.join(data_root, 'train')
        else:
            THE_PATH = os.path.join(data_root, 'val')
        data_path = []
        label = []
        for idx, folder in enumerate(os.listdir(THE_PATH)):
            this_folder_path = os.path.join(THE_PATH, folder)
            this_folder_images = os.listdir(this_folder_path)
            for image_path in this_folder_images:
                data_path.append(os.path.join(this_folder_path, image_path))
                label.append(idx)
        return data_path, label