import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
import collections
from tqdm import tqdm

# BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA10', 'QA20', 'QA60',
#          'doa', 'label', 'id']


class BreizhCropsImgTS(Dataset):
    def __init__(self, region="frh04", root="/media/maxz/Passport/Arbeit/", h5path="/media/maxz/Passport/Arbeit/frh04.h5", verbose=True):
    #def __init__(self, region, root="/tmp", h5path="/tmp", verbose=True):
        # filter_length=0, transform, target_transform, load_timeseries=True, recompile_h5_from_csv

        assert region in ["frh01", "frh02", "frh03", "frh04"]

        if verbose:
            print(f"Initializing BreizhCrops region {region}")

        self.region = region.lower()
        # self.bands = BANDS
        self.root = root
        self.year = 2018
        # year = str(self.year)
        self.level = "L1C"
        self.verbose = verbose
        self.h5path = h5path  # "/media/maxz/Passport/Arbeit/frh04.h5"
        self.classmapping = os.path.join(root, "classmapping.csv")
        self.codesfile = os.path.join(root, "codes.csv")
        self.index = pd.read_csv(os.path.join(root, "frh04.csv"))

        with h5py.File(self.h5path, "r") as dataset:
            h5ids = list(dataset[region].keys())

        self.index = self.index.loc[self.index.id.isin(h5ids)]

        cm = pd.read_csv(self.classmapping)
        codes = cm['code']
        # 0     ORH
        # 1     ORP
        # 2     BTH
        # 3     BTP
        # 4     CZH
        # 5     CZP
        # 6     MID
        # 7     MIE
        # 8     MIS
        # 9     TRN
        # 10    AGR
        # 11    PFR
        # 12    PWT
        # 13    VRG
        # 14    CAB
        # 15    CTG
        # 16    NOS
        # 17    NOX
        # 18    PIS
        # 19    PPH
        # 20    PRL
        # 21    PTR
        # 22    RGA
        self.codes = codes

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        row = self.index.iloc[index]
        label = row.CODE_CULTU
        storagepath = os.path.join(self.region, str(row.id))

        with h5py.File(self.h5path, "r") as dataset:
            listimg = list()
            listimgshp = list()
            for date in dataset[storagepath].keys():
                img = np.array(dataset[(storagepath + "/" + date + "/img")])
                mask = np.array(dataset[(storagepath + "/" + date + "/mask")])

                listimg.append(img)
                listimgshp.append(img.shape)

            listimg_ = list()
            counter = collections.Counter(listimgshp)
            c_shp = counter.most_common(1)
            for i in range(0, len(listimg)):
                if listimg[i].shape == c_shp[0][0]:
                    listimg_.append(listimg[i])

            '''
            #print(max(listimgshp))
            c_max, h_max, w_max = max(listimgshp)
            c_min, h_min, w_min = min(listimgshp)
            dH1 = (h_max - h_min)/2
            dW1 = (w_max - w_min)/2
            pad_width = [(0, 0), (int(np.floor(dH1)), int(np.ceil(dH1))), (int(np.floor(dW1)), int(np.ceil(dW1)))]

            for l in range(0, len(listimg)):
                if listimg[l].shape != max(listimgshp):
                    img_pad = np.pad(listimg[l], pad_width, mode='constant')
                    mask_pad = np.pad(mask, pad_width, mode='constant')
                else:
                    img_pad = listimg[l]
                listimgpad.append(img_pad)
            
            if mask.shape != max(listimgshp):
                mask = np.pad(mask, pad_width, mode='constant')


            for l in range(0, len(listimg)):
                print(listimg[l].shape)
            '''
            X = np.stack(listimg_)
            # '''
            t, c, h, w = X.shape
            H, W, T = 160, 160, 144
            dH = (H - h)/2
            dW = (W - w)/2
            dT = T - t
            pad_width_img = [(0, dT), (0, 0), (int(np.floor(dH)), int(np.ceil(dH))), (int(np.floor(dW)), int(np.ceil(dW)))]
            X_ = np.pad(X, pad_width_img, mode='constant')

            pad_width_mask = [(int(np.floor(dH)), int(np.ceil(dH))), (int(np.floor(dW)), int(np.ceil(dW)))]
            mask = np.pad(mask, pad_width_mask, mode='constant')
            # '''
            X = torch.from_numpy(X_)

            ids = list()
            for i in range(0, len(self.codes)):
                ids.append(self.codes[i][0])

            return X, ids, label
            # return X, mask, label


if __name__ == '__main__':
    test = BreizhCropsImgTS(region="frh04", verbose=True)
    lentest = len(test)
    test[0]

