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

    def dilate_bin_image(self, bin_image, kernel):
        """
        dilate bin image
        Args:
            bin_image: image with 0,1 pixel value
        Returns:
            dilate image
        """
        kernel_size = kernel.shape[0]
        bin_image = np.array(bin_image)
        if (kernel_size % 2 == 0) or kernel_size < 1:
            raise ValueError("kernel size must be odd and bigger than 1")
        if (bin_image.max() != 1) or (bin_image.min() != 0):
            raise ValueError("input image's pixel value must be 0 or 1")
        d_image = np.zeros(shape=bin_image.shape)
        center_move = int((kernel_size - 1) / 2)
        for i in range(center_move, bin_image.shape[0] - kernel_size + 1):
            for j in range(center_move, bin_image.shape[1] - kernel_size + 1):
                d_image[i, j] = np.max(bin_image[i - center_move:i + center_move, j - center_move:j + center_move])
        return d_image

    def _normalization(self,profile):
        smax = np.max(profile)
        smin = np.min(profile)
        profile = (profile - smin) / (smax - smin)
        return profile

    def _load_profile(self,profileDir, profileName, flag, epoch):
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
            return label
        if flag == 2:
            border_filePath = '{dir}/border/{name}'.format(dir=profileDir, name=profileName)
            border = np.array(pd.read_csv(border_filePath, header=None))

            if epoch<5:
                kernel = np.ones(shape=(11-epoch*2, 11-epoch*2))  # 无,3,5,7,9
                border = self.dilate_bin_image(border, kernel)
            return border

    def getData(self, epoch, n, batch):
        train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
        if len(train_index) == 0 or len(train_index) < batch:
            self.n = 0
            train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
        self.n += 1
        train_data = dict()
        train_data['seismic'] = []
        train_data['pha'] = []
        train_data['facies'] = []
        train_data['border'] = []
        train_data['names'] = []
        for i in train_index:
            x,pha = self._load_profile(self.data_path, i, 0, epoch)
            y= self._load_profile(self.data_path,i, 1, epoch)
            b = self._load_profile(self.data_path, i, 2, epoch)
            train_data['seismic'].append(x)
            train_data['facies'].append(y)
            train_data['pha'].append(pha)
            train_data['names'].append(i)
            train_data['border'].append(b)

        train_data['seismic'] = np.array(train_data['seismic'])
        train_data['facies'] = np.array(train_data['facies'])
        train_data['pha'] = np.array(train_data['pha'])
        train_data['border'] = np.array(train_data['border'])

        # 规定shape
        train_data['facies'].shape = (batch, self.w,self.h)
        train_data['seismic'].shape = (batch, 1, self.w,self.h)
        train_data['pha'].shape = (batch, 1, self.w,self.h)

        # 使标签为0~9
        train_data['facies'] -= 1
        return train_data

