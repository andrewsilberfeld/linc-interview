from loaders import get_train_eval_test
from typing import Literal
from torch import nn
from torch.nn import functional as F
import torch

#library that AS loaded but were available in the native yaml environment
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import os

class n_convLayers(nn.Module):

  def __init__(self,
               nb_conv_per_level: int,
               inp_channels: int,
               nb_features: int,
               conv_kernel_size: int):
    """
        create a flexibly-sized block of convolutions+ReLU

        Parameters
        ----------
        inp_channels : int
            Number of input channels
        nb_features : int
            Number of output chanels
        nb_conv_per_level : int
            Number of convolutional layers at each level.
    """

    super(n_convLayers, self).__init__()

    #instantiate one 2D convolution with specified input channels and output channels + activation
    modules = nn.ModuleList()
    modules.append(nn.Conv2d(inp_channels,nb_features,conv_kernel_size, padding = 1))
    modules.append(nn.ReLU())

    #instantiate additional 2D convolution+ReLu sequences where the 2Dconv has
    #the same number of input channels as the first conv's output channels.
    #this also becomes the number of output channels as well...
    for i in range(nb_conv_per_level-1):
      modules.append(nn.Conv2d(nb_features,nb_features,conv_kernel_size, padding = 1))
      modules.append(nn.ReLU())

    #make the list of Convolution+Relu blocks into a sequential model
    self.conv = nn.Sequential(*modules)

  def forward(self,x):
    return self.conv(x)

class Backbone(nn.Module):
    """A 2D UNet

    ```
    C -[conv xN]-> F ----------------------(cat)----------------------> 2*F -[conv xN]-> Cout
                   |                                                     ^
                   v                                                     |
                  F*m -[conv xN]-> F*m  ---(cat)---> 2*F*m -[conv xN]-> F*m
                                    |                  ^
                                    v                  |
                                  F*m*m -[conv xN]-> F*m*m
    ```
    """  # noqa: E501

    def __init__(
            self,
            inp_channels: int = 2,
            out_channels: int = 2,
            nb_features: int = 16,
            mul_features: int = 2,
            nb_levels: int = 3,
            nb_conv_per_level: int = 2,
            # Implementing the following switches is optional.
            # If not implementing the switch, choose the mode you prefer.
            activation: Literal['ReLU', 'ELU'] = 'ReLU', #AS did not parameterize the activation function -- used ReLU
            pool: Literal['interpolate', 'conv'] = 'conv', #AS changed this from 'interpolate' to 'conv', did not parameterize the pooling method
            conv_kernel_size: int = 3
            #AS added the kernel size option - AS 5/13/2025
    ):
        """
        Parameters
        ----------
        inp_channels : int
            Number of input channels
        out_channels : int
            Number of output chanels
        nb_features : int
            Number of features at the top/first UNET level
        mul_features : int
            Multiply the number of features by this number
            each time we go down one level.
        nb_levels : int
            Number of levels in the UNET
        nb_conv_per_level : int
            Number of convolutional layers at each level.
        pool : {'interpolate', 'conv'}
            Method used to go down/up one level.
            If `interpolate`, use `torch.nn.functional.interpolate`.
            If `conv`, use strided convolutions on the way down, and
            transposed convolutions on the way up.
        activation : {'ReLU', 'ELU'}
            Type of activation
        conv_kernel_size: int
            Size of one side of the square convolutional kernel

        """

        super(Backbone, self).__init__()

        #make the contracting side (down) and expanding side (up)
        contracting_modules = nn.ModuleList()
        expanding_modules = nn.ModuleList()

        #calculate the number of features per level, given nb_levels
        features = [(nb_features)*(mul_features**level) for level in range(nb_levels)]

        #create the layers of the UNET for the contracting side
        for level in range(nb_levels):
          #add the convolutional layers
          if level == 0: #level 0
            contracting_modules.append(n_convLayers(
                                          nb_conv_per_level,
                                          inp_channels,
                                          features[level],
                                          conv_kernel_size
                                          )
                                  )
          else: #levels 1 and higher
            contracting_modules.append(n_convLayers(
                                          nb_conv_per_level,
                                          features[level-1],
                                          features[level],
                                          conv_kernel_size
                                          )
                                  )
          #add maxPool after the n-convolutional layers, except for the bottom level, where an upsampling is used.
          if level != nb_levels-1: #all levels except the last one
            contracting_modules.append(nn.MaxPool2d(kernel_size = 2,stride = 2))
          else:
            contracting_modules.append(nn.ConvTranspose2d(
                                          features[level],
                                          features[level-1],
                                          kernel_size = 2,
                                          stride = 2)
                                      )
        #assign to class
        self.contracting_levels = contracting_modules

        #create the layers of the UNET for the expanding side
        for level in reversed(range(nb_levels)):
          #add the convolutional layers
          if level == nb_levels-1: #if deepest level, ignore
            """ """
          elif level == 0: #if level 0
            expanding_modules.append(n_convLayers(
                                          nb_conv_per_level,
                                          2*features[level],
                                          features[level],
                                          conv_kernel_size
                                          )
                                  )
          else: #if levels 1 to (nb_levels-1) (assuming nb_levels is at least 3)
            #add the convolutional layers halving the number of features at each level
            expanding_modules.append(n_convLayers(
                                          nb_conv_per_level,
                                          2*features[level],
                                          features[level],
                                          conv_kernel_size
                                          )
                                  )

            #add the transpose convolution to increase the resolution of the feature map
            expanding_modules.append(nn.ConvTranspose2d(
                                          features[level],
                                          features[level-1],
                                          kernel_size = 2,
                                          stride = 2)
                                      )

        #assign to class
        self.expanding_levels = expanding_modules

        #output segmentation
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, inp):
        """
        Parameters
        ----------
        inp : (B, in_channels, X, Y)
            Input tensor

        Returns
        -------
        out : (B, out_channels, X, Y)
            Output tensor
        """

        #define the contracting and expanding levels -- exclude the bottle neck for this.
        down_levels = self.contracting_levels[0:(len(self.contracting_levels)-2)] #all the down levels except the bottleneck conv and convTranspose
        bottle_neck = self.contracting_levels[-2:] #bottleneck conv and convTranspose
        up_levels = self.expanding_levels

        #define the list of data outputs for skip connections
        skip_connections = []

        #forward pass the data through the down levels
        #choose every even-idx'ed element for skip connections -- these are the contracting convolutional layers. This excludes the bottleneck standard convolution and convTranspose layers
        for idx,level in enumerate(down_levels):
            inp = level(inp) #forward pass the data through the contracting layers of the network (n_convLayer or maxpool)
            if idx%2 == 0: #if the level is an even index (the n_convLayer), then save the resulting data in a list to use as a skip connection. This excludes the bottleneck standard conv. and convTranspose layers
              skip_connections.append(inp)

        #pass the data through the bottle-neck
        inp = bottle_neck[0](inp) #standard convolutional block
        inp = bottle_neck[1](inp) #convTranspose

        #reverse the order of skip connections list so that it can be concatenated with the up_levels in order
        maxIndex = len(skip_connections)-1
        skip_connections = reversed(skip_connections)

        #pass the data through the up levels while concatenating skip connections
        for idx,skip_connect in enumerate(skip_connections):

          #concatenate the skip connection to the level
          resized_inp = F.interpolate(inp,size=skip_connect.shape[-2:], mode='bilinear') #re-size the level's data  to be the same as the skip-connections
          #note, this is differen than the original UNET which CenterCrops the skip connection to the level's size.
          #however, this enables the final output to be the same size as the original input at that level, in cases where the two are not equal.
          inp = torch.cat((skip_connect, resized_inp), 1)

          #forward pass the data through the expanding level
          n_conv = up_levels[2*idx]
          inp = n_conv(inp) #standard convolutional block

          #forward pass the up-conv block so long as we aren't at the last skip connection, for which there is no upConv block.
          if idx != maxIndex:
            upConv = up_levels[(2*idx)+1]
            inp = upConv(inp) #convTranspose

        #output the deformation field
        output = self.out(inp) #output layer

        return output

class VoxelMorph(nn.Module):
    """
    Construct a voxelmorph network with the given backbone
    """

    def __init__(self, **backbone_parameters):
        """
        Parameters
        ----------
        backbone_parameters : dict
            Parameters of the `Backbone` class
        """
        super().__init__()
        self.backbone = Backbone(2, 2, **backbone_parameters)

    def forward(self, fixmov):
        """
        Predict a displacement field from a fixed and moving images

        Parameters
        ----------
        fixmov : (B, 2, X, Y) tensor
            Input fixed and moving images, stacked along
            the channel dimension

        Returns
        -------
        disp : (B, 2, X, Y) tensor
            Predicted displacement field
        """
        return self.backbone(fixmov)

    def deform(self, mov, disp):
        """
        Deform the image `mov` using the displacement field `disp`

        Parameters
        ----------
        moving : (B, 1, X, Y) tensor
            Moving image
        disp : (B, 2, X, Y) tensor
            Displacement field

        Returns
        -------
        moved : (B, 1, X, Y) tensor
            Moved image
        """
        opt = dict(dtype=mov.dtype, device=mov.device)
        disp = disp.clone()
        nx, ny = mov.shape[-2:]

        # Rescale displacement to conform to torch conventions with
        # align_corners=True

        # 0) disp contains relative displacements in voxels
        mx, my = torch.meshgrid(torch.arange(nx, **opt),
                                torch.arange(ny, **opt), indexing='ij')
        disp[:, 0] += mx
        disp[:, 1] += my
        # 1) disp contains absolute coordinates in voxels
        disp[:, 0] *= 2 / (nx - 1)
        disp[:, 1] *= 2 / (ny - 1)
        # 2) disp contains absolute coordinates in (0, 2)
        disp -= 1
        # 3) disp contains absolute coordinates in (-1, 1)

        # Permute/flip to conform to torch conventions
        disp = disp.permute([0, 2, 3, 1])
        disp = disp.flip([-1])

        # Transform moving image
        return F.grid_sample(
            mov, disp,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

    def membrane(self, disp):
        """
        Compute the membrane energy of the displacement field
        (the average of its squared spatial gradients)
        """
        return (
            (disp[:, :, 1:, :] - disp[:, :, :-1, :]).square().mean() +
            (disp[:, :, :, 1:] - disp[:, :, :, :-1]).square().mean())

    def loss(self, fix, mov, disp, lam=0.1):
        """
        Compute the regularized loss (mse + membrane * lam)

        Parameters
        ----------
        fix : (B, 1, X, Y) tensor
            Fixed image
        mov : (B, 1, X, Y) tensor
            Moving image
        disp : (B, 2, X, Y) tensor
            Displacement field
        lam : float
            Regularization
        """
        moved = self.deform(mov, disp)
        loss = nn.MSELoss()(moved, fix)
        loss += self.membrane(disp) * lam
        return loss

def train(
      trainset,
      evalset,
      nb_epochs: int = 3,
      max_batches_per_epoch: int = None,
      learning_rate: float = 1e-03,
      calc_eval: bool = True,
      plot_loss: bool = True,
      early_stop_threshold_epoch: float = 5e-02,
      early_stop_threshold_training: float = 5e-02,
      nb_iter_perCalcEval: int = 1e03,
      vxmorph_lambda: float = 0,
      nb_features: int = 16,
      mul_features: int = 2,
      nb_levels: int = 3,
      nb_conv_per_level: int = 2,
      conv_kernel_size: int = 3):
    """
    A training function

    This function is used to generate and train a voxelmorph UNET model. 
    It returns a trained model when given the input data and optional settings for the model architecture and its training.

    Parameters
    ----------
    trainset : required training set tensor (B, 2, X, Y) tensor
        training set images
    evalset: required evaluation set tensor (B, 2, X, Y) tensor
        evaluation set images
    nb_epochs: int
        Number of epochs
    max_batches_per_epoch : int
        Maximum number of batches per epoch
	Training within an epoch often plateaus before the full training dataset is passed through the model. This is another way of shortening the training.
    learning_rate : float
        Optimizer learning rate
    calc_eval : bool
        Calculate the evaluation data during training or not
    plot_loss : bool
        Plot the real-time loss data or not
    early_stop_threshold_epoch : float
        Threshold for early stopping of the current epoch
	If the difference between the most recent evaluation loss and the average evaluation loss for the previous 4 evaluation losses within this epoch is less than this threshold, end the epoch
    early_stop_threshold_training : float
        Threshold for early stopping of the entire training process
	If the difference between the final evaluation loss for the current epoch and the average final evaluation loss for the last 4 epochs is less than this threshold, end training
    nb_iter_perCalcEval : int
        Number of batches between calculating the evaluation dataset
        Default is 1000 for efficiency
    vxmorph_lambda : float
        Regularization penalty for the membrane energy loss calculation
    nb_features : int
        Number of features at the top/first UNET level
    mul_features : int
        Multiply the number of features by this number
        each time we go down one level.
    nb_levels : int
        Number of levels in the UNET
    nb_conv_per_level : int
        Number of convolutional layers at each level
    conv_kernel_size : int
        Size of one side of the square convolutional kernel

    """
    #establish list of print statements for re-printing at completion of train
    log = []

    #if max_batchers_per_epoch is not specified, then set as the number of batches in trainset
    if max_batches_per_epoch == None:
       max_batches_per_epoch = len(trainset)

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model training on " + str(device))
    log.append("Model training on " + str(device))

    #initalize model with default parameters and initialize optimizer
    model = VoxelMorph(nb_features = nb_features,
                       mul_features = mul_features,
                       nb_levels = nb_levels,
                       nb_conv_per_level = nb_conv_per_level,
                       conv_kernel_size = conv_kernel_size).to(device) #VoxelMorph accepts the Backbone Class' parameters directly.
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    #lists for training loop
    idxs = []
    eval_idxs = []
    list_of_train_losses = []
    list_of_eval_losses = []
    colors = ['red', 'orange', 'green', 'blue','purple']

    #make the colors list longer in the case of more epochs than colors
    expand_colors_factor = int(torch.ceil(torch.tensor(nb_epochs/5)).item())
    colors = colors*expand_colors_factor 

    #Train the model with full evaluation set analyzed every n batches (nb_iter_perCalcEval)
    for epoch in range(nb_epochs):     
      train_loss = 0
      idxs.append([]) #put a list in a list for plotting the training loss data
      list_of_train_losses.append([])
      eval_idxs.append([]) #put a list in a list for plotting the eval loss data
      list_of_eval_losses.append([])

      #iterate over training image batches
      for i,train_images in enumerate(trainset):
        #set training mode
        model.train()
        optimizer.zero_grad()

        #send train images to the device
        train_images = train_images.to(device)
        fix = train_images[:,0:1,:,:].to(device)
        mov = train_images[:,1:2,:,:].to(device)

        #forward pass and calculate train loss
        disp = model(train_images).to(device)
        loss = model.loss(fix, mov, disp,lam = vxmorph_lambda)
        train_loss_batch = loss.item() * train_images.size(0) #loss (MSE) * # of images per batch
        train_loss += train_loss_batch #sum the loss from this batch with the loss from all the previous batches
        
        #backward pass and optimizer
        loss.backward()
        optimizer.step()

        #calculate a full evaluation loss for every n training batches (nb_iter_perCalcEval) if calc_eval == True
        if (calc_eval == True) and (i % nb_iter_perCalcEval == 0):
          print("Calculating eval loss")
          eval_loss = 0
          model.eval()
          with torch.no_grad():
            for eval_images in evalset:
              eval_images = eval_images.to(device)
              eval_disp = model(eval_images).to(device)
              eval_fix = eval_images[:,0:1,:,:].to(device)
              eval_mov = eval_images[:,1:2,:,:].to(device)
              loss = model.loss(eval_fix,eval_mov,eval_disp,lam = vxmorph_lambda)
              eval_loss_batch = loss.item()*eval_images.size(0)
              eval_loss += eval_loss_batch          

            #average the eval loss over all the samples [Note: to this point, the train loss has not been averaged across all samples but will be following the completion of the epoch]
            eval_loss_norm = eval_loss/(len(evalset)*eval_images.size(0))
            list_of_eval_losses[epoch].append(np.log10(eval_loss_norm))
            #get the index corresponding to the number of train batches at this point, for the specific epoch
            eval_idxs[epoch].append(i)
                
        #print training output, and if plot_loss == True, then plot it too. This Also plot the eval loss if calc_eval == True.
        if (i % 100 == 0):
          train_loss_norm = train_loss/((i+1)*train_images.size(0))
          
          if calc_eval == False:
            eval_loss_norm = "not calculated"
            print(  "epoch = " + str(epoch) + 
                    ", batch = " + str(i) + 
                    ", train_loss = " + f"{train_loss_norm:.2e}"
                  )
            log.append(  "epoch = " + str(epoch) + 
                    ", batch = " + str(i) + 
                    ", train_loss = " + f"{train_loss_norm:.2e}"
                  )
          else:
            print(  "epoch = " + str(epoch) + 
                    ", batch = " + str(i) + 
                    ", train_loss = " + f"{train_loss_norm:.2e}" +
                    ", val_loss = " + f"{eval_loss_norm:.2e}" + " (updates every " + str(nb_iter_perCalcEval) + " training batches)")
            log.append(  "epoch = " + str(epoch) + 
                    ", batch = " + str(i) + 
                    ", train_loss = " + f"{train_loss_norm:.2e}" +
                    ", val_loss = " + f"{eval_loss_norm:.2e}" + " (updates every " + str(nb_iter_perCalcEval) + " training batches)"
                
            )
            
          #plot loss vs Epoch
          if (plot_loss == True):
            #average the train loss over all the train samples calculated thus far
            idxs[epoch].append(i)
            list_of_train_losses[epoch].append(np.log10(train_loss_norm)) #epoch-specific lists of indices and log10, normalized train losses

            #plt the in-progress train loss and the evaluation loss
            plt.close()
            #plot the training data from each epoch
            for e in range(epoch+1):
              plt.scatter(idxs[e],list_of_train_losses[e], label = 'training loss, epoch = ' + str(e), c = colors[e])
              if calc_eval == True:
                plt.scatter(eval_idxs[e],list_of_eval_losses[e], label = 'eval loss, epoch = ' + str(e), c = colors[e], marker = 6)
              plt.title(str(i)+" Batches of Training")
              plt.xlabel("Batches")
              plt.ylabel("log10(Loss)")
              plt.legend()
            display.display(plt.gcf())
            display.clear_output(wait=True)
          
        #if the epoch had at least 1000 batches and the average of the last 4 eval losses are within some threshold of the newest eval loss, end this epoch
        if (calc_eval == True) and (len(list_of_eval_losses[epoch])>=5):
          last4_eval_losses = np.log10(np.mean(np.power(10,list_of_eval_losses[epoch][-5:-1])))
          mostRecent_eval_loss = list_of_eval_losses[epoch][-1]
          if (i>=1000) and (last4_eval_losses-mostRecent_eval_loss < early_stop_threshold_epoch):
            print('stopping epoch ' + str(epoch) + ' early for plateau in validation data')
            log.append('stopping epoch ' + str(epoch) + ' early for plateau in validation data')
            break

        if (max_batches_per_epoch>0) and (i >=max_batches_per_epoch):
          break

      #if the average "final eval loss" from the last 4 epochs is within some threshold of the newest epoch's final eval loss, end training
      if (calc_eval == True) and (len(list_of_eval_losses)>=5):
        last4_final_eval_losses = np.log10(np.mean(np.power(10,list_of_eval_losses[-5:-1][-1]))) #final eval loss from previous 4 epochs
        mostRecent_final_eval_loss = list_of_eval_losses[-1][-1] #final eval loss from current epoch
        if (i>=1000) and (last4_final_eval_losses-mostRecent_final_eval_loss < early_stop_threshold_training):
          print('stopping training ' + str(epoch) + ' early for plateau in validation data')
          log.append('stopping training early for plateau in validation data')
          display.clear_output(wait=False)
          break

    #print the log of all the print statements which were cleared during the graphing
    if plot_loss == True:
      print(*log, sep='\n')

    #return the model
    return model

def test(model, testset,vxmorph_lambda: float = 0):
    """
    A testing function.
    This function is used to test a previously-trained voxelmorph UNET model. 

    Parameters
    ----------
    model : required PyTorch model
    testset: required teset set tensor (B, 2, X, Y) tensor
        teset set images
    vxmorph_lambda : float
        Regularization penalty for the membrane energy loss calculation
        should be same value as used during training

    """
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model evaluating on " + str(device))

    model.eval()
    with torch.no_grad():
      test_loss = 0
      print("Running test images...\n")
      for test_images in testset:
        test_images = test_images.to(device)
        test_disp = model(test_images).to(device)
        test_fix = test_images[:,0:1,:,:].to(device)
        test_mov = test_images[:,1:2,:,:].to(device)
        loss = model.loss(test_fix,test_mov,test_disp,lam = vxmorph_lambda)
        test_loss_batch = loss.item()*test_images.size(0)
        test_loss += test_loss_batch          

      #average the test loss over all the samples [Note: to this point, the train loss has not been averaged across all samples but will be following the completion of the epoch]
      test_loss_norm = test_loss/(len(testset)*test_images.size(0))

      #print the warped images to demonstrate that moving image truly gets warped from 7 to 1
      test_moved = model.deform(test_mov,test_disp).to(device)

      images = []
      for i in range(6):
        images.append(test_fix[i,0,:,:])
        images.append(test_mov[i,0,:,:])
        images.append(test_moved[i,0,:,:])
      titles = ['Fixed Image', 'Moving Image', 'Moved Image']

      ## Plot the resulting images
      fig, axes = plt.subplots(6, 3, figsize=(5, 10))
      axes = axes.flatten()
      for i, ax in enumerate(axes):
          ax.imshow(images[i].to("cpu").detach().numpy(), cmap='gray')  # Adjust cmap as needed
          ax.set_title(titles[i%3])
          ax.axis('off')  # Hide axis ticks and labels
      plt.tight_layout()
      plt.show()

      return test_loss_norm

if __name__ == "__main__":
  #get the train/eval/test data
  trainset, evalset, testset = get_train_eval_test()

  #train and test the model
  model = train(trainset,evalset)
  test_loss = test(model,testset)

  print(f"Test_loss = {test_loss:.2e}")
  torch.save(model, os.getcwd() + "/voxelmorph_model_AndrewSilberfeld.pt") #save the model