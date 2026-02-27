import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
from PreProcessing import PreprocessData
import torch.optim as optim


# Set Seeds For Randomness
torch.manual_seed(10)
np.random.seed(10)    
InputSize = 6  # Input Size
batch_size = 1 # Batch Size Of Neural Network
NumClasses = 1 # Output Size 

############################################# FOR STUDENTS #####################################

NumEpochs = 25
HiddenSize = 10

# Create The Neural Network Model
class Net(nn.Module):
    def __init__(self, InputSize,NumClasses):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputSize, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, NumClasses)
		###### Define The Feed Forward Layers Here! ######
        
    def forward(self, x):
		###### Write Steps For Forward Pass Here! ######
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(InputSize, NumClasses)     

criterion = nn.MSELoss() ###### Define The Loss Function Here! ######
optimizer = optim.SGD(net.parameters(), lr=0.01) ###### Define The Optimizer Here! ######

##################################################################################################

if __name__ == "__main__":
        
    TrainSize,SensorNNData,SensorNNLabels = PreprocessData()   
    for j in range(NumEpochs):
        losses = 0
        for i in range(TrainSize):  
            input_values = Variable(SensorNNData[i])
            labels = Variable(SensorNNLabels[i])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
        print ('Epoch %d, Loss: %.4f' %(j+1, losses/SensorNNData.shape[0]))       
        torch.save(net.state_dict(), './SavedNets/NNBot.pkl')
           
        


