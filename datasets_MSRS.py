import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted

# Dataset
TRAIN_PATH = "/media/sata1/hedan/MSRS-main/train"
VAL_PATH = "/media/sata1/hedan/MSRS-main/test"
format_train = 'png'
format_val = 'png'


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def get_list(root_dir, img_path):
    """
    路径已经被集成在.txt里面，给根目录和文件路径就可以直接读取
    遍历MRI_filenames即获得相应位置的图像路径
    """
    MRI_list, other_list = [], []
    f = open(img_path)
    lines = f.readlines()
    for line in lines:
        line1, line2 = line.strip().split(' ')
        MRI_list.append(root_dir + line1)
        other_list.append(root_dir + line2)
    return MRI_list, other_list


class Hinet_Dataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):

        self.transform = transforms_
        self.mode = mode
        if mode == 'train':
            # train
            self.files1 = natsorted(sorted(glob.glob(TRAIN_PATH + "/ir/*")))
            self.files2 = natsorted(sorted(glob.glob(TRAIN_PATH + "/vi/*")))
        else:
            self.files1 = sorted(glob.glob(VAL_PATH + "/ir/*"))
            self.files2 = sorted(glob.glob(VAL_PATH + "/vi/*"))
            # print(VAL_PATH1 + "MRI/*." + c.format_val)
            # self.files1 = sorted(glob.glob(VAL_PATH1 + "/MRI/*." + c.format_val))
            # self.files2 = sorted(glob.glob(VAL_PATH1 + "/PET/*." + c.format_val))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files1[index])
            image = to_rgb(image)
            # image = Image.open(self.files1[index]).convert('L')
            item = self.transform(image)

            image1 = Image.open(self.files2[index])
            image1 = to_rgb(image1)
            # image1 = Image.open(self.files2[index]).convert('L')
            item1 = self.transform(image1)
            return item, item1

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        if self.mode == 'shuffle':
            return max(len(self.files_cover), len(self.files_secret))

        else:
            return len(self.files1)


transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomCrop(c.cropsize),
    T.ToTensor()
])

transform_val = T.Compose([
    T.CenterCrop(c.cropsize_val),
    T.ToTensor(),
])


# Training data loader
trainloader = DataLoader(
    Hinet_Dataset(transforms_=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)
# Test data loader
testloader = DataLoader(
    Hinet_Dataset(transforms_=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True
)