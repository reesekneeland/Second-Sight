# Find the c value with the biggest variance. 
# 


import torch
from torch.autograd import Variable
import numpy as np
from nsd_access import NSDAccess




def find_biggest_variance(output_c, target_c):
    print(list(output_c.shape))
    print(list(target_c.shape))
    
    output_c = output_c.reshape(1, 77, 1024)
    
    
    print(output_c.shape)
    
    count = 0
    for i, x in enumerate(output_c):
        for i in x:
            print(i)
            count += 1
        
    print(count)
    
    
    
    
    
    


def main():
    
    output_c = torch.load("output_c.pt")
    target_c = torch.load("target_c.pt")
    
    find_biggest_variance(output_c, target_c)


if __name__ == "__main__":
    main()