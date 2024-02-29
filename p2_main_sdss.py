#----------------------------------------------------
# Main routine for training and testing GNN models
# Author: Albert Bonnefous, adaptated from Pablo Villanueva Domingo
# Last update: 02/08/23
#----------------------------------------------------

import time, datetime, psutil
from Source.networks import *
from Source.training import *
from Source.plotting import *
from Source.create_dataset import *
from Source.constants import *
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from subprocess import Popen, run

#--- MAIN ---#

if __name__ == "__main__":
    time_ini = time.time()

    k_nn=15000
    M_gal_min = 14.0
    verbose = True
    params_dataimport = N_halo_max, M_halo_min, M_gal_min, radius_max, simdir
    params_nn = learning_rate, weight_decay, n_layers, k_nn, n_epochs, training
    setdir = "Dataset/dataset0_sdss.pt"
    outdir = "Output_sdss/"
    cmd1 = ["mkdir", "-p",  outdir]
    Popen(cmd1)

    # Import the dataset
    if verbose : print("Importing dataset from {}".format(setdir))
    dataset=torch.load(setdir)
    # Number of features
    node_features = dataset[0].x.shape[1]
    global_features = dataset[0].u.shape[1]
    if verbose :
        subs=np.sum([dataset[i].x.shape[0] for i in range(len(dataset))])//len(dataset)
        print("Total number of halos in the dataset", len(dataset), "Mean number of subhalos", subs)
    
    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(node_features, global_features, n_layers, k_nn, hidden_channels=200, latent_channels=100)
    model.to(device)
    if verbose: print("Model : " + namemodel(params_dataimport, params_nn)+"\n")
    
    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used (GB) : ", process.memory_info().rss/1.e9)
    
    # Train the net
    if training:
        time_train = time.time()
        if verbose: print("Training!\n")
        train_losses, valid_losses = training_routine(model, train_loader, valid_loader, params_dataimport, params_nn, outdir, verbose)
        if verbose : print("Training done, time elapsed for training : ",datetime.timedelta(seconds=time.time()-time_train))

    time_test = time.time()
    # Test the net
    if verbose : print("\nTesting!\n")

    # Load the trained model
    state_dict = torch.load(outdir+namemodel(params_dataimport, params_nn), map_location=device)
    model.load_state_dict(state_dict)
        
    # Test the model
    test_loss, rel_err = test(test_loader, model, params_dataimport, params_nn, outdir)
    if verbose : print("Testing done, time elapsed for testing : ",datetime.timedelta(seconds=time.time()-time_test))
    if verbose: print("Test Loss: {:.2e}".format(test_loss))

    # Correct the bias
    if verbose : print("\nCorrecting the bias !\n")
    bias = correct_bias(params_dataimport, params_nn, outdir)
    if verbose: print("Bias : {:.2e}".format(bias))

    # Test the model
    test_loss, rel_err = test(test_loader, model, params_dataimport, params_nn, outdir, bias)
    if verbose: print("Test Loss after bias correction : {:.2e}".format(test_loss))

    # Plot some graphs
    plot_outputs_vs_true(params_dataimport, params_nn, outdir)
    plot_hist_outputs_and_true(params_dataimport, params_nn, outdir)
    plot_deviation(params_dataimport, params_nn, outdir)
    if training:
        plot_losses(train_losses, valid_losses, test_loss, params_dataimport, params_nn, outdir)
    
    print("Finished. Total time elapsed : ",datetime.timedelta(seconds=time.time()-time_ini))
    
