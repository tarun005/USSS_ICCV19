import numpy as np
import os
import random

from PIL import Image
import torch

from torch.utils.data import Dataset
import glob
import sys



class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class SegmentationDataset(Dataset):
    
    def __init__(self, root, subset,
                img_path, label_path, pattern, img_suffix, label_suffix,  file_path=False, transform=None, num_images=None):


        # print(img_path)
        self.images_root = f'{root}/{img_path}/{subset}'
        self.labels_root = f'{root}/{label_path}/{subset}'
        self.image_paths = glob.glob(f'{self.images_root}/{pattern}')
        self.label_paths = [ img.replace(self.images_root, self.labels_root).replace(img_suffix, label_suffix) for img in self.image_paths  ]
        if "idd" in root:
            self.image_paths = self.image_paths[:4000]
            self.label_paths = self.label_paths[:4000]
        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]

        self.file_path = file_path
        self.transform = transform
        self.relabel = Relabel(255, self.num_classes) if transform != None else None


    def __getitem__(self, index):

        filename = self.image_paths[index]
        filenameGt = self.label_paths[index]


        with Image.open(filename) as f:
            image = f.convert('RGB')

        if self.mode == 'labeled':
            with Image.open(filenameGt) as f:
                label = f.convert('P')
        else:
            label = image

        # print(image.size, label.size)
        if self.transform !=None:
            image, label = self.transform(image, label)

        if self.d_idx == 'NYUv2_s': ## Wrap around the void class
            label = label-1
            label[label<0] = 255

        if self.relabel != None and self.mode == 'labeled':
            label = self.relabel(label)


        if self.mode == 'unlabeled':
            return image
        else:
            return image, label


    def __len__(self):
        return len(self.image_paths)


class CityscapesDataset(SegmentationDataset):

    num_classes = 19
    label_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    color_map = np.array([
        [128, 64,128],
        [244, 35,232],
        [ 70, 70, 70],
        [102,102,156],
        [190,153,153],
        [153,153,153],
        [250,170, 30],
        [220,220,  0],
        [107,142, 35],
        [152,251,152],
        [ 70,130,180],
        [220, 20, 60],
        [255,  0,  0],
        [  0,  0,142],
        [  0,  0, 70],
        [  0, 60,100],
        [  0, 80,100],
        [  0,  0,230],
        [119, 11, 32]
    ], dtype=np.uint8)


    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None , mode='labeled'):
        self.d_idx = 'CS'
        self.mode = mode
        super(CityscapesDataset, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labelTrainIds.png', transform=transform, file_path=file_path, num_images=num_images)

class ANL4Transform(object):


    def __call__(self, image, label):
        indices = label >= 30
        label[indices] = 255
        return image, label

        


class ANUEDatasetL4(SegmentationDataset):

    num_classes = 30
    label_names = ['road', 'parking', 'drivable fallback', 'sidewalk',  'non-drivable fallback', 'person', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'caravan',  'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'fallback background']

    color_map = np.array([[128, 64, 128], [250, 170, 160], [81, 0, 81], [244, 35, 232], [152, 251, 152], [220, 20, 60], [246, 198, 145], [255, 0, 0], [0, 0, 230], [119, 11, 32], [255, 204, 54], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [136, 143, 153], [220, 190, 40], [102, 102, 156], [190, 153, 153], [180, 165, 180], [174, 64, 67], [220, 220, 0], [250, 170, 30], [153, 153, 153], [0, 0, 0], [70, 70, 70], [150, 100, 100], [107, 142, 35], [70, 130, 180], [169, 187, 214]], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None):
        self.d_idx = 'ANUE'
        super(ANUEDatasetL4, self).__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel4Ids.png', transform=transform, file_path=file_path, num_images=num_images)



class IDD_Dataset(SegmentationDataset):

    num_classes = 26
    label_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'animal', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky']

    color_map   = np.array([
        [128, 64, 128], #road
        [ 81,  0, 81], #drivable fallback
        [244, 35, 232], #sidewalk
        [152, 251, 152], #nondrivable fallback
        [220, 20, 60], #pedestrian
        [255, 0, 0],  #rider
        [0, 0, 230], #motorcycle
        [119, 11, 32], #bicycle
        [255, 204, 54], #autorickshaw
        [0, 0, 142], #car
        [0, 0, 70], #truck
        [0, 60, 100], #bus
        [136, 143, 153], #vehicle fallback
        [220, 190, 40], #curb
        [102, 102, 156], #wall
        [190, 153, 153], #fence
        [180, 165, 180], #guard rail
        [174, 64, 67], #billboard
        [220, 220, 0], #traffic sign
        [250, 170, 30], #traffic light
        [153, 153, 153], #pole
        [169, 187, 214], #obs-str-bar-fallback
        [70, 70, 70], #building
        [150, 120, 90], #bridge
        [107, 142, 35], #vegetation
        [70, 130, 180] #sky
    ], dtype=np.uint8)

    def __init__(self, root, subset='train', transform=None, file_path=False, num_images=None, mode='labeled'):
        self.d_idx = 'IDD'
        self.mode = mode
        super().__init__(root, subset,  
                img_path = 'leftImg8bit', label_path='gtFine', pattern='*/*',
                img_suffix = '_leftImg8bit.png' , label_suffix='_gtFine_labellevel3Ids.png', transform=transform, file_path=file_path, num_images=num_images)



class CamVid(SegmentationDataset):

    num_classes = 11
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'CVD'
        self.mode = mode


        self.images_root = f"{root}/{subset}/"
        self.labels_root = f"{root}/{subset}annot/"
        

        self.image_paths = glob.glob(f'{self.images_root}/*.png')
        self.label_paths = glob.glob(f'{self.labels_root}/*.png')

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class SunRGB(SegmentationDataset):

    num_classes = 37
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'SUN'
        self.mode = mode

        listname = f"{root}/{subset}37.txt"

        with open(listname , 'r') as fh:
            self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        with open(listname , 'r') as fh:
            self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None

class NYUv2_seg(SegmentationDataset):

    num_classes = 13
    # label_names = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"]
    # color_map = np.array([64,128,64], [192,0,128], [0,128,192], [0,128,64], [128,0,0], [64,0,128], [64,0,192], [192,128,64], [192,192,128], [64,64,128], [128,0,192], [192,0,64], [128,128,64], [192,0,192], [128,64,64], [64,192,128], [64,64,0], [128,64,128], [128,128,192], [0,0,192], [192,128,128], [128,128,128], [64,128,192], [0,0,64], [0,64,64], [192,64,128], [128,128,0], [192,128,192], [64,0,64], [192,192,0], [0,0,0], [64,192,0])
    

    def __init__(self, root, subset='train', transform=None,  file_path=False, num_images=None, mode="labeled"):

        self.d_idx = 'NYU_s'
        self.mode = mode

        # listname = f"{root}/{subset}13.txt"

        images = os.listdir(os.path.join(root , subset , 'images'))
        labels = os.listdir(os.path.join(root , subset , 'labels'))

        self.image_paths = [f"{root}/{subset}/images/"+im_id for im_id in images]
        self.label_paths = [f"{root}/{subset}/labels/"+lb_id for lb_id in labels]

        # with open(listname , 'r') as fh:
        #     self.image_paths = [os.path.join(root , l.split()[0]) for l in fh]

        # with open(listname , 'r') as fh:
        #     self.label_paths = [os.path.join(root , l.split()[-1]) for l in fh]

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.label_paths = self.label_paths[:num_images]
            
        self.file_path = file_path
        self.transform = transform

        self.relabel = Relabel(255, self.num_classes) if transform != None else None


def colorize(img, color, fallback_color=[0,0,0]): 
    img = np.array(img)
    W,H = img.shape
    view = np.tile(np.array(fallback_color, dtype = np.uint8), (W,H, 1) )
    for i, c in enumerate(color):
        indices = (img == i)
        view[indices] = c
    return view


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    

    def show_data(ds):
        print(len(ds))
        import random
        i = random.randrange(len(ds))
        img, gt = ds[i]
        color_gt = colorize(gt, ds.color_map)
        print(img.size,color_gt.shape)
        plt.imshow(img)
        plt.imshow(color_gt, alpha=0.25)
        plt.show()


    # cs = CityscapesDataset('/ssd_scratch/cvit/girish.varma/dataset/cityscapes')
    # show_data(cs)

    # an = ANUEDataset('/ssd_scratch/cvit/girish.varma/dataset/anue')
    # show_data(an)

    # bd = BDDataset('/ssd_scratch/cvit/girish.varma/dataset/bdd100k')
    # show_data(bd)

    # mv = MVDataset('/ssd_scratch/cvit/girish.varma/dataset/mvd')
    # show_data(mv)
