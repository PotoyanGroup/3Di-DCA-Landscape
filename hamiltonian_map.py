import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from model.generator import seq_code, read_fasta_as_one_hot_encoded
from dca.dca_class import dca
import time
import pickle

### Requires src-python-mfdca package from EIL github ###
#
# Produces MSA of grid sequences, saved as 'grid_msa.fasta'
# Produces numpy array of form [x,y,Hamiltonian], saved as grid_dataset.pkl
# Produces pixel rendered plot saved as pixel_plot.png
#

class Generative_Functions:
    def __init__(self, model,output_path,output_pkl_path,full_alignment,pixels=500):
        self.model=model
        self.output_path=output_path
        self.output_pkl_path=output_pkl_path
        self.full_alignment=full_alignment
        self.pixels=pixels
        self.loaded = load_model(self.model,compile=True)
    
    def get_axis_values(self) -> int:
        ids=[]
        ds = tf.data.Dataset.from_generator(lambda: read_fasta_as_one_hot_encoded(self.full_alignment),tf.int8).batch(1000)
        _,_,zed = self.loaded.encoder.predict(ds)    
        largest_value = max(abs(zed.min()),abs(zed.max()))
        square_max = int(np.ceil(largest_value))
        self.axis_min = -square_max # x/y min
        self.axis_max = square_max # x/y max


    def get_key(val):
        for key, value in seq_code.items():
            if val == value:
                return key


    def return_sequence(self,latent_output):
        seq = ''.join(self.get_key(x) for x in np.argsort(latent_output, axis=0)[-1,:])
        return seq


    def make_grid_msa(self, batch_size=10000):
        start = time.time()
        self.get_axis_values()
        resolution = [self.axis_min, self.axis_max, self.pixels]
        trained_model = load_model(self.model_path)
        sampling_set = np.linspace(resolution[0], resolution[1], resolution[2])
        a = np.meshgrid(sampling_set, sampling_set)
        coord = np.vstack(np.array(a).transpose())
        with open(self.output_path, 'w') as fd:
            for batch_idx in range(0,coord.shape[0],batch_size):
                if batch_idx+batch_size > coord.shape[0]: #bigger than array
                    z_input = coord[batch_idx:]
                else:
                    z_input = coord[batch_idx:batch_idx+batch_size]
                latent_output = trained_model.decoder.predict(z_input)
                sequences = [self.return_sequence(seq_mat) for seq_mat in latent_output]
                for idx_seq, (x, y) in enumerate(z_input):
                    fd.writelines('> ' + str(x) + ' ' + str(y) + '\n')
                    fd.writelines(sequences[idx_seq] + '\n')
        fd.close()
        end = time.time()

        print('Time Elapsed - '+str(end-start))
        self.coord=coord


    def get_hamiltonian(self):
        start = time.time()
        self.get_axis_values()
        resolution = [self.axis_min, self.axis_max, self.pixels]
        sampling_set = np.linspace(resolution[0], resolution[1], resolution[2])
        a = np.meshgrid(sampling_set, sampling_set)
        self.coord = np.vstack(np.array(a).transpose())
        mfdcamodel = dca(self.full_alignment)
        mfdcamodel.mean_field()

        grid_hamiltonians, _ = mfdcamodel.compute_Hamiltonian(self.output_path)

        grid_for_plotter = np.zeros((self.coord.shape[0],3))
        grid_for_plotter[:,:2] = self.coord
        grid_for_plotter[:,2] = grid_hamiltonians

        grid_hamiltonians = grid_hamiltonians.reshape(resolution[2],resolution[2])
        end = time.time()
        print('Time Elapsed - '+str(end-start))
        return grid_hamiltonians, grid_for_plotter,mfdcamodel


    def plot_hamil_latent(self,hamil, title):
        resolution = [self.axis_min, self.axis_max, self.pixels]
        # generate 2 2d grids for the x & y bounds
        a, b = np.meshgrid(np.linspace(resolution[0], resolution[1], resolution[2]),
                        np.linspace(resolution[0], resolution[1], resolution[2]))

        fig, ax = plt.subplots()
        c = ax.pcolormesh(b, a, hamil, cmap='jet')
        ax.set_title(title)
        fig.colorbar(c, ax=ax)
        plt.show()
