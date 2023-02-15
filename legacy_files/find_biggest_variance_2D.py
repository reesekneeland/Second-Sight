import torch
from torch.autograd import Variable
import itertools


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


def find_biggest_variance_2D_clip_vector():

    # Store the indexes
    # Store values of the clip vectors at those points and save them to a 73000x100 tensor, sorted by highest variance on the 100 dimension
    # Make a dictionary

    top_clip_vectors = torch.empty((73000, 100))

    # Load the flattened clip vectors. 
    matrix = torch.empty((73000, 78848))
    for i in range(73000):
        matrix[i] = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/c/" + str(i) + ".pt").reshape((1,78848))


    tensor_variances = {}
    

    for index in range(78848):

        # Iterate over each column
        column = matrix[:, index]

        # Calculate the varaince of that column
        current_variance = torch.var(column)

        # Store the index and the variance in a dictonary
        tensor_variances[index] = current_variance

    # Sort the dictionary 
    sorted_tensor_varainces= sorted(tensor_variances.items(), key=lambda x:x[1], reverse=True)
    tensor_variances_dict = dict(sorted_tensor_varainces)

    # Take the top 100 variances of the dictioanry
    tensor_variances_top_hundred = dict(itertools.islice(tensor_variances_dict.items(), 100))
    #print("Dictionary limited by K is : " + str(tensor_variance_top_hundred))

    varRank = 0

    # Iterate over the 73000 clip vectors 
    for img in range(73000):
        varRank = 0

        # Add the values to the top clip vecetor tensor. 
        for key, value in tensor_variances_top_hundred.items():
                top_clip_vectors[img, varRank] = matrix[img, key]
                varRank += 1

    # Save off the tensor
    torch.save(top_clip_vectors, "top_hundred_varinace_clip_vector.pt")
    
    
def find_biggest_variance_2D_z_vector():

    # Store the indexes
    # Store values of the z (Brain Data) vectors at those points and save them to a 73000x100 tensor, sorted by highest variance on the 100 dimension
    # Make a dictionary

    top_z_vectors = torch.empty((73000, 100))

    # Load the flattened z vectors. 
    matrix = torch.empty((73000, 16384))
    for i in range(73000):
        torch_tensor_z = torch.load("/home/naxos2-raid25/kneel027/home/kneel027/nsd_local/nsddata_stimuli/tensors/z/" + str(i) + ".pt").reshape((1,16384))
        matrix[i] = torch_tensor_z


    tensor_variances = {}

    for index in range(16384):

        # Iterate over each column
        column = matrix[:, index]

        # Calculate the varaince of that column
        current_variance = torch.var(column)

        # Store the index and the variance in a dictonary
        tensor_variances[index] = current_variance

    # Sort the dictionary 
    sorted_tensor_varainces= sorted(tensor_variances.items(), key=lambda x:x[1], reverse=True)
    tensor_variances_dict = dict(sorted_tensor_varainces)

    # Take the top 100 variances of the dictioanry
    tensor_variances_top_hundred = dict(itertools.islice(tensor_variances_dict.items(), 100))

    varRank = 0

    # Iterate over the 73000 z vectors 
    for img in range(73000):
        varRank = 0

        # Add the values to the top z vecetor tensor. 
        for key, value in tensor_variances_top_hundred.items():
                top_z_vectors[img, varRank] = matrix[img, key]
                varRank += 1

    # Save off the tensor
    torch.save(top_z_vectors, "top_hundred_varinace_z_vector.pt")



def main():

    find_biggest_variance_2D_z_vector()


if __name__ == "__main__":
    main()