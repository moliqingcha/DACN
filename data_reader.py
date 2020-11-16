
import h5py
import numpy as np

class H5DataLoader(object):

    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        data_file = h5py.File(data_path, 'r')
        self.images, self.labels = data_file['X'], data_file['Y']
        self.gen_indexes()
    
    def gen_indexes(self):
        if self.is_train:
            self.indexes = np.random.permutation(range(self.images.shape[0]))
        else:
            self.indexes = np.array(range(self.images.shape[0]))
        self.cur_index = 0
    
    def next_batch(self, batch_size):
        next_index = self.cur_index+batch_size
        cur_indexes = list(self.indexes[self.cur_index:next_index])
        self.cur_index = next_index
        
        if len(cur_indexes) < batch_size and self.is_train:
            self.gen_indexes()
            self.cur_index = batch_size
            cur_indexes = list(self.indexes[:batch_size])
       
        if len(cur_indexes)==0 and not self.is_train:
            cur_indexes = [0]
        elif len(cur_indexes) < batch_size and not self.is_train:
            self.cur_index = 0
        cur_indexes.sort()
        
        outx, outy = self.images[cur_indexes], self.labels[cur_indexes]
        
        return outx, outy
