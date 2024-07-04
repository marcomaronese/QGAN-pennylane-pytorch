import numpy as np
import random
import torch

def tensor_BAS(num_samples, dim_x, dim_y):
    """
    tensor_BAS fa sample di immagini dal dataset Bars and Stripes e le restituisce
    sotto forma di tensori pytorch.

    Args:
        num_samples:    numero di immagini da produrre
        dim_x:          numero di pixel lungo asse orizzontale
        dim_y:          numero di pixel lungo asse verticale

    """
    set_dim = 2 **dim_x + 2 ** dim_y
    sample_elements = np.ones(
    (num_samples, dim_x, dim_y)) * 0.5
    for index in range(num_samples):
        i=0
        # La prossima riga forse non è il massimo teorico ma esclude le immagini
        # "tutte 1" e "tutte -1", qualsiasi sia la dimensione x e y
        while i==0 or i== 2**dim_x-1 or i==2**dim_x or i==set_dim-1:
            i = random.choice(range(set_dim))
        if i < 2 ** dim_x:
            j = i
            for column in range(dim_x):
                a = j % 2
                j = ((j - a) / 2)
                if a == 0:
                    sample_elements[index, column, :] = np.ones(dim_y)
                else:
                    sample_elements[index, column, :] = -np.ones(dim_y)
        else:
            j = i - 2 ** dim_x
            for line in range(dim_y):
                a = j % 2
                j = ((j - a) / 2)
                if a == 0:
                    sample_elements[index, :, line] = np.ones(dim_x)
                else:
                    sample_elements[index, :, line] = -np.ones(dim_x)
    final_samples = sample_elements.reshape(sample_elements.shape[0], sample_elements.shape[1] * sample_elements.shape[2])  
    return torch.Tensor(final_samples)