import torch
from torch.nn.modules.activation import Tanh
from GAN import GAN 
from Discriminator import Discriminator
from utils import tensor_BAS
from time import time
import Generators
import matplotlib.pyplot as plt
import os
import parameters
import datetime

def main():
    """
    main() del programma
    """

    # Import dei parametri
    epochs =                parameters.values.get("epochs")
    batches_per_epoch =     parameters.values.get("batches_per_epoch")
    batch_size =            parameters.values.get("batch_size")
    dim_x =                 parameters.values.get("dim_x")
    dim_y =                 parameters.values.get("dim_y")
    generator_latent_dim =  parameters.values.get("generator_latent_dim")
    isquantum =             parameters.values.get("isquantum")
    lr_d =                  parameters.values.get("learning_rate_discriminator")
    lr_g =                  parameters.values.get("learning_rate_generator")
    discriminator_pause =   parameters.values.get("discriminator_pause")

    data_dimension = dim_x*dim_y

    # Definizione generatore: classico o quantistico
    if isquantum == 0:
        generator = Generators.Classic_Generator(latent_dim=generator_latent_dim,
        output_dim=data_dimension, output_activation=Tanh())
    if isquantum == 1:
        generator = Generators.Quantum_Generator(data_dimension, qlayers=2, shots=1)

    discriminator = Discriminator(data_dimension, [8, 8, 1])
    
    # Definizione della funzione noise, che genera l'input per il generatore
    noise_fn = lambda x: torch.rand((x, generator_latent_dim), device='cpu')

    # Definizione della funzione data, che genera vettori estratti casualmente
    #  dal dataset Bars and Stripes
    data_fn = lambda x: tensor_BAS(x, dim_x, dim_y)

    # Definizone GAN
    gan = GAN(generator, discriminator, noise_fn, data_fn, batch_size,
     lr_d=lr_d, lr_g=lr_g)

    loss_g, loss_d_real, loss_d_fake = [], [], []
    
    start = time()
    dataT =datetime.datetime.now().strftime("D%d-%m_T%H-%M-%S")
    print(f"Learning start at time {dataT}")
    os.mkdir('run_' + dataT)

    # Questo ciclo scorre le epoche e allena la GAN
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0, 0, 0
        count_discriminator_number_of_steps = 0
        for dummy_index in range(batches_per_epoch):
            if dummy_index % discriminator_pause==0:
                """ Training step per il discriminatore """

                """
                Non avviene tutti gli step. Questo perché può accadere, e
                ci è parso accadesse in questo caso, che il processo di apprendimento
                per il discriminatore sia troppo semplice. Ciò risulta in un
                discriminatore troppo severo, e la conseguente impossibilità per il
                generatore di apprendere.
                """
                ldr_, ldf_ = gan.train_step_discriminator()
                loss_d_real_running += ldr_
                loss_d_fake_running += ldf_
                count_discriminator_number_of_steps += 1
            
            """ Training step per il generatore """
            lg_ = gan.train_step_generator()
            loss_g_running += lg_
        # mediamo le loss raccolte negli step del batch
        loss_g.append(loss_g_running / batches_per_epoch)
        loss_d_real.append(loss_d_real_running / count_discriminator_number_of_steps)
        loss_d_fake.append(loss_d_fake_running / count_discriminator_number_of_steps)
        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
              f" G={loss_g[-1]:.3f},"
              f" Dr={loss_d_real[-1]:.3f},"
              f" Df={loss_d_fake[-1]:.3f}")
        if epoch%20==0:
            print(gan.generate_samples(num=10))

if __name__ == "__main__":
    main()