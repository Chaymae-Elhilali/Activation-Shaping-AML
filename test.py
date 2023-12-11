import torch

def get_M_random_generator_function(alpha):
    #Input: tensor size; output: random 
    size = [100]
    M = torch.ones(size)
    M = torch.where(torch.rand(size) <= alpha, M, torch.zeros(size))
    return M

def count_n_ones(M):
    return torch.sum(M)

for i in range(100):
    M = get_M_random_generator_function(0.1)
    print(count_n_ones(M)/100)