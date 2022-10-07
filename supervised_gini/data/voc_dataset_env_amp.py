import numpy as np
import pandas as pd
import getFeature

# class YQYvocDataSet():
#     def __init__(self, data_path,label_path,name_path,seed=None):
#         self.data_path = data_path
#         self.label_path = label_path
#         self.name_path = name_path
#         self.w = 590
#         self.h = 1006
#         self.data_ids = [i_id.strip() for i_id in open(name_path)]
#         self.n = 0
#         if seed != None:
#             np.random.seed(seed)
#             np.random.shuffle(self.data_ids)
#
#     def _load_profile(self,profileDir, profileName, flag):
#         filePath = '{dir}/{name}'.format(dir=profileDir, name=profileName)
#         profile = np.array(pd.read_csv(filePath,header=None))
#         if flag == 0:
#             smax = np.max(profile)
#             smin = np.min(profile)
#             profile = (profile - smin) / (smax - smin)
#         profile.shape = (1, self.w,self.h)
#         return profile
#
#     def getData(self, n, batch):
#         train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
#         if len(train_index) == 0 or len(train_index) < batch:
#             self.n = 0
#             train_index = self.data_ids[self.n * batch:(self.n + 1) * batch]
#         self.n += 1
#         train_data = dict()
#         train_data['seismic'] = []
#         train_data['facies'] = []
#         for i in train_index:
#             x = self._load_profile(self.data_path, i, 0)
#             y = self._load_profile(self.label_path,i, 1)
#             train_data['seismic'].append(x)
#             train_data['facies'].append(y)
#
#         train_data['seismic'] = np.array(train_data['seismic'])
#         train_data['facies'] = np.array(train_data['facies'])
#
#         # 规定shape
#         train_data['facies'].shape = (batch, self.w,self.h)
#         train_data['seismic'].shape = (batch, 1, self.w,self.h)
#
#         # 使标签为0~9
#         train_data['facies'] -= 1
#         return train_data
#
# class YQYvocDataSet_unlabel():
#     def __init__(self, data_path,label_path,name_path,seed):
#         self.data_path = data_path
#         self.label_path = label_path
#         self.name_path = name_path
#         self.w = 590
#         self.h = 1006
#         self.data_ids = [i_id.strip() for i_id in open(name_path)]
#         np.random.seed(seed)
#         np.random.shuffle(self.data_ids)
#
#     def _load_profile(self,profileDir, profileName, flag):
#         filePath = '{dir}/{name}'.format(dir=profileDir, name=profileName)
#         profile = np.array(pd.read_csv(filePath,header=None))
#         if flag == 0:
#             smax = np.max(profile)
#             smin = np.min(profile)
#             profile = (profile - smin) / (smax - smin)
#         profile.shape = (1, self.w,self.h)
#         return profile
#
#
#     def getData(self,n, batch):
#         train_index = self.data_ids[n * batch:(n + 1) * batch]
#
#         train_data = dict()
#         train_data['seismic'] = []
#         train_data['facies'] = []
#         for i in train_index:
#             x = self._load_profile(self.data_path, i, 0)
#             y = self._load_profile(self.label_path, i, 1)
#             train_data['seismic'].append(x)
#             train_data['facies'].append(y)
#
#         train_data['seismic'] = np.array(train_data['seismic'])
#         train_data['facies'] = np.array(train_data['facies'])
#
#         # 规定shape
#         train_data['facies'].shape = (batch,  self.w,self.h)
#         train_data['seismic'].shape = (batch, 1,  self.w,self.h)
#
#         # 使标签为0~9
#         train_data['facies'] -= 1
#         return train_data


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
            env_filePath = '{dir}/env/{name}'.format(dir=profileDir, name=profileName)
            amp_filePath = '{dir}/amp/{name}'.format(dir=profileDir, name=profileName)

            seismic = np.array(pd.read_csv(seismic_filePath,header=None))
            env = np.array(pd.read_csv(env_filePath,header=None))
            amp = np.array(pd.read_csv(amp_filePath,header=None))

            seismic = self._normalization(seismic)
            env = self._normalization(env)
            amp = self._normalization(amp)

            env.shape = (1, self.w, self.h)
            amp.shape = (1, self.w, self.h)
            seismic.shape = (1,self.w,self.h)

            return seismic,env,amp
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
        train_data['env'] = []
        train_data['amp'] = []
        train_data['facies'] = []
        train_data['names'] = []
        for i in train_index:
            x,env,amp = self._load_profile(self.data_path, i, 0)
            y= self._load_profile(self.data_path,i, 1)
            train_data['seismic'].append(x)
            train_data['facies'].append(y)
            train_data['env'].append(env)
            train_data['amp'].append(amp)
            train_data['names'].append(i)

        train_data['seismic'] = np.array(train_data['seismic'])
        train_data['facies'] = np.array(train_data['facies'])
        train_data['env'] = np.array(train_data['env'])
        train_data['amp'] = np.array(train_data['amp'])

        # 规定shape
        train_data['facies'].shape = (batch, self.w,self.h)
        train_data['seismic'].shape = (batch, 1, self.w,self.h)
        train_data['env'].shape = (batch, 1, self.w,self.h)
        train_data['amp'].shape = (batch, 1, self.w,self.h)

        # 使标签为0~9
        train_data['facies'] -= 1
        return train_data


class YQYvocDataSet_Feature_unlabel():
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
            env_filePath = '{dir}/env/{name}'.format(dir=profileDir, name=profileName)
            amp_filePath = '{dir}/amp/{name}'.format(dir=profileDir, name=profileName)

            seismic = np.array(pd.read_csv(seismic_filePath,header=None))
            env = np.array(pd.read_csv(env_filePath,header=None))
            amp = np.array(pd.read_csv(amp_filePath,header=None))

            seismic = self._normalization(seismic)
            env = self._normalization(env)
            amp = self._normalization(amp)

            env.shape = (1, self.w, self.h)
            amp.shape = (1, self.w, self.h)
            seismic.shape = (1,self.w,self.h)

            return seismic,env,amp
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
        train_data['env'] = []
        train_data['amp'] = []
        for i in train_index:
            x,env,amp = self._load_profile(self.data_path, i, 0)
            train_data['seismic'].append(x)
            train_data['env'].append(env)
            train_data['amp'].append(amp)

        train_data['seismic'] = np.array(train_data['seismic'])
        train_data['env'] = np.array(train_data['env'])
        train_data['amp'] = np.array(train_data['amp'])

        # 规定shape
        train_data['seismic'].shape = (batch, 1, self.w,self.h)
        train_data['env'].shape = (batch, 1, self.w,self.h)
        train_data['amp'].shape = (batch, 1, self.w,self.h)

        return train_data