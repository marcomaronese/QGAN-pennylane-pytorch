import torch
import torch.optim as optim
from torch import nn

"""
Implementazione di un Generative Adversarial Network

Metodi:
GAN.__init__(): 
    inizializzazione della classe

GAN.generate_samples(): 
    helper function per ottenere sample random dal generatore.
        
        
GAN.train_step_generator(): 
    Fa uno step di training sul Generatore e restituisce la loss.

GAN.train_step_discriminator(): 
    Fa uno step di training sul Discriminatore e restituisce la loss.
"""


class GAN():
    def __init__(self, generator, discriminator, noise_fn, data_fn,
                 batch_size, lr_d, lr_g, device='cpu',):
        """Classe per allenare Generatore e Discriminatore
            Args:
                generator:      oggetto Generatore
                discriminator:  oggetto Discriminatore
                noise_fn:       funzione noise che genera il rumore che verrà
                                mappato nei samples
                data_fn:        funzione che restituisce i dati di training
                batch_size:     minibatch size
                device:         cpu or cuda (default cpu)
                lr_d:           learning rate discriminator
                lr_g:           learning rate generator
        """
        self.generator = generator
        self.generator = self.generator.to(device)
        self.discriminator = discriminator
        self.discriminator = self.discriminator.to(device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.device = device
        """ La loss function è la binary cross-entropy """
        self.criterion = nn.BCELoss()
        """ Gli ottimizzatori sono entrambi Adam """
        self.optim_d = optim.Adam(discriminator.parameters(),
                                  lr=lr_d, betas=(0.7, 0.99))
        self.optim_g = optim.Adam(generator.parameters(),
                                  lr=lr_g, betas=(0.7, 0.99))
        """ Le seguenti sono le label per il training """
        self.target_ones = torch.ones((batch_size, 1)).to(device)
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)

    def generate_samples(self, latent_vec=None, num=None):
        """ Funzione per samplare dal generatore
            Args:
                latent_vec: (optional) tensore 2d PyTorch contenente vettori
                             latenti da fornire come input al generatore.
                             Se non è definito, viene chiamata la funzione
                             noise per generarlo.
                num:        (optional) numero sample da produrre. Se non è 
                             definito, num = batch_size.
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        """ torch.no_grad() segnala di non tenere traccia dei gradienti in questa
        computazione (infatti non vogliamo fare backpropagation qui) """
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        """ Allena il generatore e restituisce la loss """
        
        """ torch.zero_grad() annula i gradienti. Lo usiamo perché così
        potranno accumularsi da zero per lo step di allenamento."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size) #sample dalla noise-generating function
        generated = self.generator(latent_vec)      # Generatore al lavoro, viene generato 
                                                    # sottobanco il grafo computazionale del
                                                    # generatore


        classifications = self.discriminator(generated) # Discriminatore al lavoro
        """ A questo punto il grafo computazionale del discriminatore è attaccato
         a quello del generatore. Nella prossima riga calcoliamo la loss
         per il generatore. """
        loss = self.criterion(classifications, self.target_ones)
        """  loss.backward() fa il gradiente della loss rispetto ad ogni parametro trainabile
        nel grafo computazionale. """
        loss.backward()
        self.optim_g.step()
        """ optim_g modifica i parametri; ma solo quelli del generatore, perché
        glieli abbiamo dati come variabile alla sua creazione. """

        return loss.item()
        

    def train_step_discriminator(self):
        """ Allena il discriminatore e restituisce la loss """
        self.discriminator.zero_grad()

        # real samples
        real_samples = self.data_fn(self.batch_size)
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        """ no_grad() perché non voglio trainare il generatore """
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        """ combino i computational graphs per i sample veri e falsi """
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()