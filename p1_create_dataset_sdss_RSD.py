#----------------------------------------------------
# Main routine for creating the training, testing and validation dataset
# Author: Albert Bonnefous, adapted from Pablo Villanueva Domingo
# Last update: 02/08/23
#----------------------------------------------------

import time, datetime, psutil
from Source.constants import *
import numpy as np
import time, datetime, psutil
import random
from decimal import Decimal
from joblib import Parallel, delayed
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from astropy.cosmology import WMAP7 as cosmo

boxsize = 2688000 # [kpc]
zsim = 0.47; 
Hz = cosmo.H(zsim).value
h100 = cosmo.H0.value/100.

# Split training, validation and testing sets, create the data loader
def split_datasets(dataset):
    random.shuffle(dataset)

    num_train = len(dataset)
    split_valid = int(np.floor(valid_size * num_train))
    split_test = split_valid + int(np.floor(test_size * num_train))

    train_dataset = dataset[split_test:]
    valid_dataset = dataset[:split_valid]
    test_dataset = dataset[split_valid:split_test]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

# Center all the origins of the coordinates in the dataset at the position of the halo
def center_halo(pos_halo,tab,box_size):
    #halo : position of the halo, (3)
    #tab : tab of the pos, mass of the galaxies, maybe some other datas (n,>=3)
    tab_c=np.copy(tab)
    tab_c[:,:3] = tab_c[:,:3]-pos_halo
    modif=(tab_c[:,:3]<0)*(2*tab_c[:,:3]+box_size<0)
    tab_c[:,:3][modif]+=box_size
    modif=(tab_c[:,:3]>0)*(2*tab_c[:,:3]-box_size>0)
    tab_c[:,:3][modif]-=box_size
    return tab_c

# Take all the galaxies in radius_max of the center halo and create one halo set
def create_halo_set(tab,pos_halo,y_halo,i,radius_max,box_size):
    #center the galaxies around the halo
    tab_c=center_halo(pos_halo[i],tab,box_size)
    #take only the galaxies within a sphere of +/- radius_max pc around the halo
    gdist = np.sqrt( np.sum(tab_c[:,:3]**2, axis=1) )
    keep = (gdist < radius_max)
    tab_c = tab_c[keep]
    
    if len(tab_c)==0: return None
    else :
    #create the global quantities
        u = np.zeros((1,2), dtype=np.float32)
        u[0,0] = tab_c.shape[0]                  # number of subhalos
        u[0,1] = np.sum(tab_c[:,-1])             # total galactic mass
        y = y_halo[i]                             # v_x, m, v
        # Create the graph of the halo
        # x: features (includes positions), pos: positions, y: vz, u: global quantity (number of subhalos)
        graph = Data(x=torch.tensor(tab_c, dtype=torch.float32), pos=torch.tensor(tab_c[:,:3], dtype=torch.float32), y=torch.tensor(y, dtype=torch.float32), u=torch.tensor(u, dtype=torch.float))
        return graph

#--MAIN--
if __name__ == "__main__":
    
    time_ini = time.time()
    
    simdir = '/home/balbert/data/sims/Magneticum/box0/'
    savedir="Dataset/dataset0_sdss_RSD.pt" 
    M_gal_min = None
    M_halo_min = None
    
    verbose = True
    params_dataimport = N_halo_max, M_halo_min, M_gal_min, radius_max, simdir

    time_import = time.time()
    if verbose : print("Importing data from "+simdir)
    
    x_gal, y_gal, z_gal, vx_gal, vy_gal, vz_gal, u_gal, V_gal, g_gal, r_gal, i_gal, Z_gal, Y_gal, J_gal, H_gal, K_gal, L_gal, M_gal, Mstar_gal, SFR_gal = np.loadtxt(simdir+'Box0_mr_bao_025_gal_cat.dat.gz', skiprows=1, unpack=True)

    ### RSD
    z_rsd = (1+zsim)*(vz_gal)/Hz * 1e3 # [kpc/h]
    z_gal_rsd = (z_gal + z_rsd + boxsize)%boxsize
    ###

    pos_gal = np.array([x_gal, y_gal, z_gal_rsd]).T
    v_gal = np.array([vx_gal, vy_gal, vz_gal]).T
    if M_gal_min!=None :
        keep_M_gal = (Mstar_gal > M_gal_min)
        pos_gal = pos_gal[keep_M_gal]
        v_gal = v_gal[keep_M_gal]
        Mstar_gal = Mstar_gal[keep_M_gal]
    if verbose : print("Total number of galaxies : ",len(x_gal))
    if verbose and M_gal_min!=None: print("Total number of galaxies over {:.1e}".format(Decimal(M_gal_min)) + " Msol : ",len(pos_gal[:,0]))

    x_halo, y_halo, z_halo, vx_halo, vy_halo, vz_halo, M_500c_halo, Tm_500_halo, Lx_500_halo = np.loadtxt(simdir+'Box0_mr_bao_025_cl_cat.dat.gz', skiprows=1, unpack=True)

    ### RSD
    zh_rsd = (1+zsim)*(vz_halo)/Hz * 1e3 # [kpc/h]
    z_halo_rsd = (z_halo + zh_rsd + boxsize)%boxsize
    ###

    pos_halo = np.array([x_halo, y_halo, z_halo_rsd]).T
    v_halo = np.array([vx_halo, vy_halo, vz_halo]).T
    if M_halo_min!=None:
        keep_M_halo = (M_500c_halo > M_halo_min)
        pos_halo = pos_halo[keep_M_halo]
        v_halo = v_halo[keep_M_halo]
        M_500c_halo = M_500c_halo[keep_M_halo]
    
    if verbose : print("Total number of halo : ",len(x_halo))
    if verbose and M_halo_min!=None : print("Total number of halo over {:.1e}".format(Decimal(M_halo_min)) + " Msol : ",len(pos_halo[:,0]))

    if verbose : print("Time elapsed for data importation : {}\n".format(datetime.timedelta(seconds=time.time()-time_import)))
    if verbose : print("Creating the dataset")
    
    #box_size=np.amax(x_gal)    
    box_size=boxsize    
    if N_halo_max==None: N_halo = len(pos_halo[:,0])
    else: N_halo=min(N_halo_max,len(pos_halo[:,0]))
    indexes = np.random.choice(len(pos_halo[:,0]),N_halo, replace=False)
    time_dataset = time.time()

    tab = np.column_stack((pos_gal,Mstar_gal, SFR_gal))
    y_halo = v_halo[:,2]    # v_z

    dataset = Parallel(n_jobs=30)(delayed(create_halo_set)(tab,pos_halo,y_halo,i,radius_max,box_size) for i in indexes)
    dataset = [a for a in dataset if a!=None]

    if verbose : print("Dataset created")
    if verbose :
        subs=np.sum([dataset[i].x.shape[0] for i in range(len(dataset))])//len(dataset)
        print("Total number of halos in the dataset", len(dataset), "Mean number of subhalos", subs)
    
    if verbose : print("Time elapsed for dataset creation : {} \n".format(datetime.timedelta(seconds=time.time()-time_dataset)))

    torch.save(dataset,savedir)
    print("Dataset saved in {}\n".format(savedir))
    
    print("Finished. Total time elapsed : ",datetime.timedelta(seconds=time.time()-time_ini))
