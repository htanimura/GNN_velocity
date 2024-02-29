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
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

k_nn=15000
M_gal_min = 14.0
verbose = True
params_dataimport = N_halo_max, M_halo_min, M_gal_min, radius_max, simdir
params_nn = learning_rate, weight_decay, n_layers, k_nn, n_epochs, training

outdir = "Output_sdss_RSD/"
outputs = np.load(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+"_biascorr.npy")
trues = np.load(outdir+"trues_"+namemodel(params_dataimport,params_nn)+"_biascorr.npy")

simdir = '/home/balbert/data/sims/Magneticum/box0/'
x_halo, y_halo, z_halo, vx_halo, vy_halo, vz_halo, M_500c_halo, Tm_500_halo, Lx_500_halo = np.load(simdir+'Box0_mr_bao_025_gal_cat.dat_1e14.npy')

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

    return ax

# Plot loss trends
def plot_losses(train_losses, valid_losses, test_loss, params_dataimport, params_nn):
    epochs = params_nn[4]
    fig, ax = plt.subplots()
    plt.plot(range(epochs), train_losses, "r-",label="Training")
    plt.plot(range(epochs), valid_losses, ":", color="indigo",label="Validation")
    #plt.yscale("log")
    plt.legend()
    plt.title(f"Test loss : {test_loss:.2e}")
    plt.savefig("Plots/loss_"+namemodel(params_dataimport,params_nn)+".png", bbox_inches='tight', dpi=300)
    plt.close()
    return None

# Plot the graphs of the GNN outputs vs the true values
def plot_outputs_vs_true(params_dataimport, params_nn):
    # Load true values and predicted means and standard deviations
    
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

    return None


# Plot the graphs of the difference between the GNN outputs and the true values, vs the true values
def plot_deviation(params_dataimport, params_nn,bias=None):
    # Load true values and predicted means and standard deviations

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
    
    return None

# Plot the histogram of the GNN outputs and of the true values
def plot_hist_outputs_and_true(params_dataimport, params_nn):
    # Load true values and predicted means and standard deviations

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
    
    return None



def plot_corr(x1,y1, bias=1.0, xlab=None, ylab=None, lab=None, fit=False, lim=1500, ps=0.003):

    y1 = y1/bias
    
    # fitting with line
    def fit_scatter(x):
        return (np.std(y1/x)-np.std(x1))**2
    
    res = minimize_scalar(fit_scatter)
    fit = res.x
    
    slope = np.std(y1)/np.std(x1)
    def model(x, b): 
        mod = slope*x + b
        return mod
    
    initial_guess = [0]
    vflag = (200 < np.abs(x1))
    #vflag3 = (200 < np.abs(Vr_true3)) * (np.abs(Vr_true3) < 1000)
    popt, pcov = curve_fit(model, x1[vflag], y1[vflag], p0=initial_guess)
    xx = np.linspace(-1000,1000,100)
    yy = slope*xx + popt[0]
    print("slope=%.2f, int=%.1f" %(slope, popt[0]))                

    
    fig = plt.figure()
    #fig = plt.figure(figsize=(6.5, 5))
    #x2 = Vr_lin2 
    #y2 = Vr_CNN2
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    
    # the scatter plot:
    density_scatter(x1,y1,fig,ax_scatter,bins=30,s=0.5)

    ax_scatter.set_xlabel('$%s \, (km/s)$' %(xlab))
    ax_scatter.set_ylabel('$%s \, (km/s)$' %(ylab))                
    
    # now determine nice limits by hand:
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    if(fit==True):

        if(popt[0]>0):
            ax_scatter.plot(xx,yy, 'r--', label='Fit $(%s=%.2f %s + %.1f)$' %(ylab,slope,xlab,popt[0]))
        else:
            ax_scatter.plot(xx,yy, 'r--', label='Fit $(%s=%.2f %s %.1f)$' %(ylab,slope,xlab,popt[0]))

    ax_scatter.plot([-lim,lim],[-lim,lim], 'b--', label='$%s = %s$' %(ylab,xlab))
    ax_scatter.legend(loc='upper left', prop={'size':12})
    ax_scatter.locator_params(axis='both', nbins=4)
    
    bins = np.linspace(-1000,1000,30) 
    ax_histx.hist(x1, bins=bins, color='k', histtype='step')
    ax_histy.hist(y1, bins=bins, color='k', histtype='step', orientation='horizontal')
    
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histx.set_yticklabels([])
    ax_histy.set_xticklabels([])

    plt.tight_layout()

    return fit



plot_corr(trues, outputs, xlab='\it{v}_{true}', ylab='\it{v}_{GNN}')
print("Uncertainty:", np.std(outputs - trues))
print("Corr coef:", np.corrcoef(outputs, trues))
