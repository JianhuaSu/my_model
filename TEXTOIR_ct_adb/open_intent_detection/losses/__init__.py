from .CosineFaceLoss import CosineFaceLoss
from .SupConLoss import SupConLoss
from torch import nn 

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'Binary_CrossEntropyLoss': nn.BCELoss(),
                'CosineFaceLoss': CosineFaceLoss(),
                'SupConLoss': SupConLoss()
            }
