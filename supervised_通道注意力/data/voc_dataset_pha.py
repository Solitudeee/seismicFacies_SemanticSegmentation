import numpy as np
import pandas as pd

class YQYvocDataSet_Feature():
    def __init__(self, data_path,name_path,seed=None):
        self.data_path = data_path
        self.name_path = name_path
        self.w = 590
        self.h = 1006
        self.data_ids = [i_id.strip() for i_id in open(name_path)]
        self.n = 0
        if seed != None:
            np.random.seed(seed)
            np.random.shuffle(self.data_ids)

    def _normalization(self,profile):
        smax = np.max(profile)
        smin = np.min(profile)
        profile = (profile - smin) / (smax - smin)
        return profile

    def _load_profile(self,profileDir, profileName, flag):
        if flag == 0:
            seismic_filePath = '{dir}/seismic/{name}'.format(dir=profileDir, name=profileName)
            pha_filePath = '{dir}/pha/{name}'.format(dir=profileDir, name=profileName)

            seismic = np.array(pd.read_csv(seismic_filePath,header=None))
            pha = np.array(pd.read_csv(pha_filePath,header=None))

            seismic = self._normalization(seismic)
            pha = self._normalization(pha)

            pha.shape = (1, self.w, self.h)
            seismic.shape = (1,self.w,self.h)

            return seismic,pha
        if flag == 1:
            label_filePath = '{dir}/label/{name}'.format(dir=profileDir, name=profileName)
            label = np.array(pd.read_csv(label_filePath, header=None))

            label.shape = (self.w, self.h)
            return label


    def getData(self, n, batch):
        train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
        if len(train_index) == 0 or len(train_index) < batch:
            self.n = 0
            train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
        self.n += 1
        train_data = dict()
        train_data['seismic'] = []
        train_data['pha'] = []
        train_data['facies'] = []
        train_data['names'] = []
        for i in train_index:
            x,pha = self._load_profile(self.data_path, i, 0)
            y= self._load_profile(self.data_path,i, 1)
            train_data['seismic'].append(x)
            train_data['facies'].append(y)
            train_data['pha'].append(pha)
            train_data['names'].append(i)

        train_data['seismic'] = np.array(train_data['seismic'])
        train_data['facies'] = np.array(train_data['facies'])
        train_data['pha'] = np.array(train_data['pha'])

        # 规定shape
        train_data['facies'].shape = (batch, self.w,self.h)
        train_data['seismic'].shape = (batch, 1, self.w,self.h)
        train_data['pha'].shape = (batch, 1, self.w,self.h)

        # 使标签为0~9
        train_data['facies'] -= 1
        return train_data

