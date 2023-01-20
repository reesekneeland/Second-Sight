# Find the c value with the biggest variance. 



import torch
from torch.autograd import Variable
import itertools

# def find_biggest_variance(output_c, target_c):
#     print(list(output_c.shape))
#     print(list(target_c.shape))
    
#     output_c = output_c.reshape(1, 77, 1024)
#     target_c = target_c.reshape(1, 77, 1024)
    
    
#     print(output_c.shape)
#     print(target_c.shape)
    
#     max_variance = 0
#     index_max_varaince = 0

#     for i, x in enumerate(output_c):
#         for j, output_sample in enumerate(x):
            
#             current_varince = torch.var(output_sample)
#             if(current_varince > max_variance):
#                 max_variance = current_varince
#                 index_max_varaince = j 
    
#     print(max_variance)
#     print(index_max_varaince)
#     print(output_c[0][index_max_varaince])
                

def find_biggest_variance_2D():

    # Store the indexes
    # Store values of the clip vectors at those points and save them to a 73000x100 tensor, sorted by highest variance on the 100 dimension
    # Make a dictionary
    
    # Create an empty tensor for the column values with the biggest variance. 
    top_clip_vectors = torch.empty((73000, 100))

    # Store the flattened clip vectors into a tensor by loading them in one by one. 
    matrix = torch.empty((73000, 78848))
    for i in range(73000):
        matrix[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c/" + str(i) + ".pt").reshape((1,78848))


    tensor_variances = {}
        
    for index in range(78848):

        # Grab the column of the matrix
        column = matrix[:, index]

        # Calculate the varaince for that column
        current_variance = torch.var(column)

        # Store that index and column variance
        tensor_variances[index] = current_variance

    # Sort the diction based on variance. 
    sorted_tensor_varainces= sorted(tensor_variances.items(), key=lambda x:x[1], reverse=True)
    tensor_variances_dict = dict(sorted_tensor_varainces)

    # Take the top 100 values of the dictionary only
    tensor_variances_top_hundred = dict(itertools.islice(tensor_variances_dict.items(), 100))
    #print("Dictionary limited by K is : " + str(tensor_variance_top_hundred))


    # Iterate over every image in the tensor
    for img in range(73000):
        varRank = 0

        # Loop over the dictionary and fill in the tensor values one by one 
        # into the top clip vector tensor. 
        for key, value in tensor_variances_top_hundred.items():
                top_clip_vectors[img, varRank] = matrix[img, key]
                varRank += 1

    # Save off the tensor. 
    torch.save(top_clip_vectors, "top_hundred_varinace_clip_vector.pt")



def main():

    find_biggest_variance_2D()


if __name__ == "__main__":
    main()