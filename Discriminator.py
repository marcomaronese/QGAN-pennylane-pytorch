import torch
from torch import nn

"""
Il discriminatore eredita dalla classe nn.Module, che è la classe 
base per le reti neurali in pytorch.

Metodi:

Discriminator.__init__(): chiama super, che da alla nostra funzione tutti
i metodi e le proprietà di nn.Module. In più ci aggiungiamo
i vari layer. In questo caso lo scopo è renderlo flessibile,
infatti i layer sono da passare come variabili.

Discriminator._init_layers(): poteva essere messo in __init__()
ma in questo modo c'è una separazione logica più ordinata. La funzione
compone i layer aggiungendo una Leaky ReLU dopo ognuno, e aggiungendo
una Sigmoid dopo l'ultimo.

Discriminator.forward(): calcolo forward della rete. Sfrutta la lista
già creata in _init_layers()
"""

class Discriminator(nn.Module):
    def __init__(self, input_dim, layers):
        """Il discriminatore cerca di discernere dati reali da dati artefatti
        Args:
            input_dim (int):    Dimensioni dell'input
            layers (List[int]): Lista delle dimensioni dei layer 
                                 (inclusa dimensione output)
        All'output applichiamo una Sigmoide (adatta perché devo generare 
         una previsione probabilistica).
        """
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self._init_layers(layers)

    def _init_layers(self, layers):
        """Inizializza i layer e li archivia come self.module_list."""
        self.module_list = nn.ModuleList()
        last_layer = self.input_dim
        for index, width in enumerate(layers):
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU())
            else:
                self.module_list.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass: mappa i sample alla probabilità che siano reali """
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate