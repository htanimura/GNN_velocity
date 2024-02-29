#----------------------------------------------------------------------
# Routines for training and testing the GNNs
# Author: Albert Bonnefous, adapted from Pablo Villanueva Domingo
# Last update: 02/08/23
#----------------------------------------------------------------------

from scipy.spatial.transform import Rotation as Rot
from Source.constants import *
import random

# use GPUs if available
if torch.cuda.is_available():
    print("\nCUDA Available\n")
    device = torch.device('cuda')
else:
    print("\nCUDA Not Available\n")
    device = torch.device('cpu')

# Training step over the whole training data
def train(loader, model, optimizer):
    model.train()
    if data_augmentation :
        #rotmat = Rot.random().as_matrix()
        theta=random.uniform(0,np.pi)
        rotmat = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training dataset.
        # Rotate randomly for data augmentation
        if data_augmentation :
            data.pos = torch.tensor(np.array([rotmat.dot(p) for p in data.pos]), dtype=torch.float32)
            data.x[:,:3] = torch.tensor(np.array([rotmat.dot(p) for p in data.x[:,:3]]), dtype=torch.float32)

        data.to(device)
        optimizer.zero_grad()                   # Clear gradients.
        out = model(data)                       # Perform a single forward pass.
        y_out = out[:,0]    # Take mean and standard deviation of the output

        loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
        #loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
        loss = loss_mse

        loss.backward()     # Derive gradients.
        optimizer.step()    # Update parameters based on gradients.
        loss_tot += loss.item()

    return loss_tot/len(loader)

# Testing/validation step
def test(loader, model, params_dataimport, params_nn, outdir, bias=None):
    model.eval()
    outs = np.zeros((1))
    trues = np.zeros((1))
    yerrors = np.zeros((1))

    errs = []
    loss_tot = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():
            data.to(device)
            out = model(data)  # Perform a single forward pass.
            
            # Correct the bias
            if bias!=None : out = out*bias
            
            y_out = out[:,0]
            err = (y_out.reshape(-1) - data.y)/data.y
            errs.append( np.abs(err.detach().cpu().numpy()).mean(axis=0) )

            loss_mse = torch.mean((y_out - data.y)**2 , axis=0)
            #loss_lfi = torch.mean(((y_out - data.y)**2 - err_out**2)**2, axis=0)
            loss = loss_mse
      
            loss_tot += loss.item()

            # Append true values and predictions
            outs = np.append(outs, y_out.detach().cpu().numpy(), 0)
            trues = np.append(trues, data.y.detach().cpu().numpy(), 0)
            #yerrors = np.append(yerrors, err_out.detach().cpu().numpy(), 0)

    # remove the initial 0
    outs = outs[1:]
    trues = trues[1:]
    
    #yerrors = yerrors[1:]
    # Save true values and predictions
    #np.save("Outputs/outputs_"+namemodel(params_dataimport,params_nn)+".npy",outs)
    #np.save("Outputs/trues_"+namemodel(params_dataimport,params_nn)+".npy",trues)
    #np.save("Outputs/errors_"+namemodel(params_dataimport,params_nn)+".npy",yerrors)

    if bias!=None : 
        np.save(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+"_biascorr.npy",outs)
        np.save(outdir+"trues_"+namemodel(params_dataimport,params_nn)+"_biascorr.npy",trues)
    else:
        np.save(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+".npy",outs)
        np.save(outdir+"trues_"+namemodel(params_dataimport,params_nn)+".npy",trues)

    return loss_tot/len(loader), np.array(errs).mean(axis=0)

# Training procedure
def training_routine(model, train_loader, valid_loader, params_dataimport, params_nn, outdir, verbose=True):

    learning_rate, weight_decay, n_layers, k_nn, n_epochs, training = params_nn

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, valid_losses = [], []
    valid_loss_min = 10**30
    err_min = 10**30

    # Training loop
    for epoch in range(1, n_epochs+1):
        train_loss = train(train_loader, model, optimizer)
        valid_loss, err = test(valid_loader, model, params_dataimport, params_nn, outdir)
        train_losses.append(train_loss); valid_losses.append(valid_loss)

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {valid_loss:.2e}')
        # Save model if it has improved
        if valid_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,valid_loss))
            #torch.save(model.state_dict(), "Models/"+namemodel(params_dataimport, params_nn))
            torch.save(model.state_dict(), outdir+namemodel(params_dataimport, params_nn))
            valid_loss_min = valid_loss
            err_min = err

    return train_losses, valid_losses

# Return the linear bias between the true values and the GNN outputs so that the two distributions match
def correct_bias(params_dataimport, params_nn, outdir):
    #outputs = np.load("Outputs/outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    #trues = np.load("Outputs/trues_"+namemodel(params_dataimport,params_nn)+".npy")
    outputs = np.load(outdir+"outputs_"+namemodel(params_dataimport,params_nn)+".npy")
    trues = np.load(outdir+"trues_"+namemodel(params_dataimport,params_nn)+".npy")

    deviation_trues = np.sqrt(np.mean((trues-np.mean(trues))**2))
    deviation_outputs = np.sqrt(np.mean((outputs-np.mean(outputs))**2))
    return deviation_trues/deviation_outputs

    
