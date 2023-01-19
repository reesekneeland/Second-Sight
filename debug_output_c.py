# Find the c value with the biggest variance. 



import torch
from torch.autograd import Variable

def find_biggest_variance(output_c, target_c):
    print(list(output_c.shape))
    print(list(target_c.shape))
    
    output_c = output_c.reshape(1, 77, 1024)
    target_c = target_c.reshape(1, 77, 1024)
    
    
    print(output_c.shape)
    print(target_c.shape)
    
    max_variance = 0
    index_max_varaince = 0

    for i, x in enumerate(output_c):
        for j, output_sample in enumerate(x):
            
            current_varince = torch.var(output_sample)
            if(current_varince > max_variance):
                max_variance = current_varince
                index_max_varaince = j 
    
    print(max_variance)
    print(index_max_varaince)
    print(output_c[0][index_max_varaince])
                
    # for i, x in enumerate(target_c):
    #     for j, output_sample in enumerate(x):
            
    #         current_varince = torch.var(output_sample)
    #         if(current_varince > max_variance):
    #             max_variance = current_varince
    #             index_max_varaince = j 
                
    # print(max_variance)
    # print(index_max_varaince)
    # print(target_c[0][index_max_varaince])


def main():
    
    #output_c = torch.load("output_c.pt")
    output_c = torch.load("output_c_3D_flattened.pt")
    target_c = torch.load("target_c.pt")
    
    find_biggest_variance(output_c, target_c)


if __name__ == "__main__":
    main()