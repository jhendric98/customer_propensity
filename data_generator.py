import numpy as np
import pickle
import keras
import h5py 

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filename, train_val='train', n_classes=2, batch_num=-1, batch_size=32, 
                 scalers=None, dim=(32,32,32), n_channels=1, shuffle=True):
        'Initialization'
        self.filename = filename
        self.train_val = train_val
        self.dim = dim        
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.scalers = scalers
        self.batch_num = batch_num
        self.getbatches()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.idx_batch) / self.batch_size))

    def getbatches(self):
        'Gets a list of batch keys available in the h5py file'
        with h5py.File(self.filename, 'r') as hdf:
            self.batches = sorted(list(hdf.keys()))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[ index*self.batch_size:(index+1)*self.batch_size ]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.batch_num += 1
        if self.batch_num >= len(self.batches):
            self.batch_num = 0
        self.batch = self.batches[self.batch_num]

        with h5py.File(self.filename, 'r') as hdf:
            self.X_batch = np.array(hdf[self.batch][self.train_val]['X'])
            self.y_batch = np.array(hdf[self.batch][self.train_val]['y'])
            self.idx_batch = np.array(hdf[self.batch][self.train_val]['idx'])
        if self.scalers is not None:
            self.X_batch = self.scale_data(self.X_batch, self.scalers)
               
        self.indexes = np.arange(len(self.idx_batch))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def scale_data(self, X, scalers):
        'Scale X data'
        X_ss = np.copy(X)            
        
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                X_ss[:, i, j, :] = scalers[i][j].transform(X[:, i, j, :])
        return X_ss    
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, idx in enumerate(indexes):
            # Store sample
            X[i,] = self.X_batch[idx]

            # Store class
            y[i,] = self.y_batch[idx]

        return X, y