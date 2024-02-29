#----------------------------------------------------------------------
# List of constants and some common functions
# Author: Albert Bonnefous, adaptated from Pablo Villanueva Domingo
# Last update: 10/11/21
#----------------------------------------------------------------------

import numpy as np
import torch
import os
import random

#--- PARAMETERS AND CONSTANTS ---#

# Reduced Hubble constant
hred = 0.7

N_halo_max = None        #max number of halo in the dataset, None for the whole set
#M_halo_min = 10**13     #min mass of the halos in the dataset, None for the whole set
M_halo_min = None     #min mass of the halos in the dataset, None for the whole set
M_gal_min = None           #min mass of the galaxies in the dataset, None for the whole set
radius_max = 100000      #max distance for the galaxies considered in a given halo graph
simdir = '/home/balbert/data/sims/Magneticum/box2b/'

# Validation and test size
valid_size, test_size = 0.15, 0.15

#learning_rate = 1e-4
#weight_decay = 1e-6
#n_layers = 3
#n_epochs = 30

# v35
#learning_rate = 1e-3
#weight_decay = 1e-4
#n_layers = 2
#n_epochs = 30
#k_nn = 15000
#training = True
#batch_size = 8

# v37
learning_rate = 1e-3
weight_decay = 1e-6
n_layers = 3
#n_epochs = 20
k_nn = 15000
#k_nn = 18100
training = True
batch_size = 32
#batch_size = 16

# v39
n_epochs = 30

# v38 True is worse
#data_augmentation = True
data_augmentation = False

#--- FUNCTIONS ---#

# Name of the model and hyperparameters
def namemodel(params_dataimport, params_nn):
    N_halo_max, M_halo_min, M_gal_min, radius_max, simdir = params_dataimport
    learning_rate, weight_decay, n_layers, k_nn, n_epochs, training = params_nn
    name="Magneticum_lr_{:.2e}_weightdecay_{:.2e}_layers_{:d}_knn_{:.2e}_epochs_{:d}_radius_{:d}".format(learning_rate, weight_decay, n_layers, k_nn, n_epochs,radius_max)
    if M_gal_min!=None:
        name=name+"_Mgalmin"
    return name

