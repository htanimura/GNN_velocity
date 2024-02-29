#----------------------------------------------------------------------
# Script for plotting some statistics
# Author: Albert Bonnefous
# Last update: 02/08/23
#----------------------------------------------------------------------

import matplotlib.pyplot as plt
from Source.constants import *
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from decimal import Decimal
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
mpl.rcParams.update({'font.size': 12})

def density_scatter( x , y, fig= None, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

# Plot loss trends
def plot_losses(train_losses, valid_losses, test_loss, params_dataimport, params_nn, outdir):
    epochs = params_nn[4]
    fig, ax = plt.subplots()
    plt.plot(range(epochs), train_losses, "r-",label="Training")
    plt.plot(range(epochs), valid_losses, ":", color="indigo",label="Validation")
    #plt.yscale("log")
    plt.legend()
    plt.title(f"Test loss : {test_loss:.2e}")
    #plt.savefig("Plots/loss_"+namemodel(params_dataimport,params_nn)+".png", bbox_inches='tight', dpi=300)
    plt.savefig(outdir+"loss_"+namemodel(params_dataimport,params_nn)+".png", bbox_inches='tight', dpi=300)
    plt.close()
    return None

# Plot the graphs of the GNN outputs vs the true values
def plot_outputs_vs_true(params_dataimport, params_nn, outdir):
    # Load true values and predicted means and standard deviations
    #outputs = np.load("Outputs/outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    #trues = np.load("Outputs/trues_"+namemodel(params_dataimport,params_nn)+".npy")
    outputs = np.load(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    trues = np.load(outdir+"trues_"+namemodel(params_dataimport,params_nn)+".npy")
    
    N_halo_max, M_halo_min, M_gal_min, radius_max, simdir = params_dataimport
    learning_rate, weight_decay, n_layers, k_nn, n_epochs, training = params_nn
    fig, ax = plt.subplots()
    
    # Plot predictions vs true values
    truemin, truemax = trues.min(), trues.max()
    ax.plot([truemin,truemax],[truemin,truemax],"r:")
    density_scatter(trues,outputs,fig,ax,bins=100,s=0.5)
    
    # Name, indicating which are the training and testing suites:
    namefig = "scatter_out_true_"+namemodel(params_dataimport, params_nn)

    # Legend, labels, etc
    plt.ylabel(r"$v_{GNN} (km.s-1)$")
    plt.xlabel(r"$v_{true} (km.s-1)$")
    mse = np.mean((outputs - trues)**2)
    textleg='\n'.join(("r_max = {:.1e}".format(Decimal(radius_max)),
                       "lr = {}".format(learning_rate),
                       "weight_decay = {:.1e}".format(weight_decay),
                       "k_nn = {:.1e}".format(Decimal(k_nn)),
                       "n_layers = {}".format(n_layers),
                       "n_epochs = {}\n".format(n_epochs)))
    if N_halo_max!=None: textleg="N_halo = {:.1e}".format(Decimal(N_halo_max))+textleg
    else : textleg = "N_halo = all\n"+textleg
    if M_gal_min!=None: textleg="M_gal_min={:.1e}\n".format(Decimal(M_gal_min))+textleg
    else : textleg="M_gal_min=None\n"+textleg
    if M_halo_min!=None:textleg="M_halo_min={:.1e}\n".format(Decimal(M_halo_min))+textleg
    else : textleg="M_halo_min=None\n"+textleg
    if data_augmentation : textleg=textleg+"Data_augmentation\n"
    textleg = textleg+"\nmse = {:.1e}".format(mse)
    ax.text(0.05, 0.95,textleg,c="black",transform=ax.transAxes, fontsize=6,verticalalignment='top')
    
    #fig.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    fig.savefig(outdir+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close()
    return None

# Plot the graphs of the difference between the GNN outputs and the true values, vs the true values
def plot_deviation(params_dataimport, params_nn, outdir, bias=None):
    # Load true values and predicted means and standard deviations
    #outputs = np.load("Outputs/outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    #trues = np.load("Outputs/trues_"+namemodel(params_dataimport,params_nn)+".npy")
    outputs = np.load(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    trues = np.load(outdir+"trues_"+namemodel(params_dataimport,params_nn)+".npy")

    if bias!=None : outputs = outputs*bias
    
    N_halo_max, M_halo_min, M_gal_min, radius_max, simdir = params_dataimport
    learning_rate, weight_decay, n_layers, k_nn, n_epochs, training = params_nn
    fig, ax = plt.subplots()
    
    # Plot predictions vs true values
    truemin, truemax = trues.min(), trues.max()
    ax.plot([truemin,truemax],[0,0],"r:")
    density_scatter(trues,outputs-trues,fig,ax,bins=100,s=0.5)
    
    # Name, indicating which are the training and testing suites:
    namefig = "error_out_true_"+namemodel(params_dataimport, params_nn)

    # Legend, labels, etc
    plt.ylabel(r"$v_{GNN} - v_{true} (km.s-1)$")
    plt.xlabel(r"$v_{true} (km.s-1)$")
    mse = np.mean((outputs - trues)**2)
    textleg='\n'.join(("r_max = {:.1e}".format(Decimal(radius_max)),
                       "lr = {}".format(learning_rate),
                       "weight_decay = {:.1e}".format(weight_decay),
                       "k_nn = {:.1e}".format(Decimal(k_nn)),
                       "n_layers = {}".format(n_layers),
                       "n_epochs = {}\n".format(n_epochs)))
    if N_halo_max!=None: textleg="N_halo = {:.1e}".format(Decimal(N_halo_max))+textleg
    else : textleg = "N_halo = all\n"+textleg
    if M_gal_min!=None: textleg="M_gal_min={:.1e}\n".format(Decimal(M_gal_min))+textleg
    else : textleg="M_gal_min=None\n"+textleg
    if M_halo_min!=None:textleg="M_halo_min={:.1e}\n".format(Decimal(M_halo_min))+textleg
    else : textleg="M_halo_min=None\n"+textleg
    if data_augmentation : textleg=textleg+"Data_augmentation\n"
    textleg = textleg+"\nmse = {:.1e}".format(mse)
    ax.text(0.05, 0.95,textleg,c="black",transform=ax.transAxes, fontsize=6,verticalalignment='top')
    
    #fig.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    fig.savefig(outdir+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close()
    return None

# Plot the histogram of the GNN outputs and of the true values
def plot_hist_outputs_and_true(params_dataimport, params_nn, outdir):
    # Load true values and predicted means and standard deviations
    #outputs = np.load("Outputs/outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    #trues = np.load("Outputs/trues_"+namemodel(params_dataimport,params_nn)+".npy")
    outputs = np.load(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    trues = np.load(outdir+"trues_"+namemodel(params_dataimport,params_nn)+".npy")

    N_halo_max, M_halo_min, M_gal_min, radius_max, simdir = params_dataimport
    learning_rate, weight_decay, n_layers, k_nn, n_epochs, training = params_nn
    fig, ax = plt.subplots()
    
    # Plot histogram of the predictions vs true values
    truemin, truemax = trues.min(), trues.max()
    ax.hist(trues,color="red", bins=100,histtype='step',label=r"$v_{true} (Mpc)$")
    ax.hist(outputs,color="indigo", bins=100,histtype='step',label=r"$v_{GNN} (Mpc)$")
    
    deviation_trues = np.sqrt(np.mean((trues)**2))
    deviation_outputs = np.sqrt(np.mean((outputs)**2))
   
    # Name, indicating which are the training and testing suites:
    namefig = "hist_out_true_"+namemodel(params_dataimport, params_nn)

    # Legend, labels, etc
    plt.xlabel(r"$v_z (km.s-1)$")
    mse = np.mean((outputs - trues)**2)
    textleg='\n'.join(("r_max = {:.1e}".format(Decimal(radius_max)),
                       "lr = {}".format(learning_rate),
                       "weight_decay = {:.1e}".format(weight_decay),
                       "k_nn = {:.1e}".format(Decimal(k_nn)),
                       "n_layers = {}".format(n_layers),
                       "n_epochs = {}\n".format(n_epochs)))
    if N_halo_max!=None: textleg="N_halo = {:.1e}".format(Decimal(N_halo_max))+textleg
    else : textleg = "N_halo = all\n"+textleg
    if M_gal_min!=None: textleg="M_gal_min={:.1e}\n".format(Decimal(M_gal_min))+textleg
    else : textleg="M_gal_min=None\n"+textleg
    if M_halo_min!=None:textleg="M_halo_min={:.1e}\n".format(Decimal(M_halo_min))+textleg
    else : textleg="M_halo_min=None\n"+textleg
    if data_augmentation : textleg=textleg+"Data_augmentation\n"
    textleg = textleg+"\nmse = {:.1e}".format(mse)
    textvar="Standard deviation : \ntrues {}\noutputs {}".format(round(deviation_trues,3),round(deviation_outputs,3))
    textleg=textleg+'\n'+textvar
    ax.text(0.05, 0.95,textleg,c="black",transform=ax.transAxes, fontsize=6,verticalalignment='top')
    plt.legend()
    
    #fig.savefig("Plots/"+namefig+".png", bbox_inches='tight', dpi=300)
    fig.savefig(outdir+namefig+".png", bbox_inches='tight', dpi=300)
    plt.close()
    return None
