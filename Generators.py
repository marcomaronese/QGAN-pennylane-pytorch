import torch
from torch import nn
import numpy as np
import pennylane as qml

"""
Il generatore eredita dalla classe nn.Module, che è la classe 
base per le reti neurali in pytorch.

Metodi:

Generator.__init__(): chiama super, che da alla nostra funzione tutti
i metodi e le proprietà di nn.Module. 

Nel caso classico componiamo i vari layer mentre per il caso 
quantistico dobbiamo definire dei layer quantistici e quindi creare 
un circuito quantistico e convertirlo in un layer di pytorch usando
il metodo qml.qnn.TorchLayer().

Generator.forward():  funzione fondamentale per le classi figlie
di nn.Module. Definisce la struttura del network.
"""

class Classic_Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, output_activation=None):
        """Generatore classico
        Args:
            latent_dim (int):   Dimensione del "noise vector"
            layers (List[int]): Lista delle dimensioni dei layer 
                                 (inclusa dimensione output)
            output_activation:  Funzione di attivazione pytorch per
                                 l'ultimo layer (oppure None)
        """
        super(Classic_Generator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 32)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, output_dim)
        self.output_activation = output_activation

    def forward(self, input_tensor):
        """ Forward pass: Mappa dal latent space (vettore noise) 
        al sample space (immagine da generare) """
        intermediate = self.linear1(input_tensor)
        intermediate = self.tanh(intermediate)
        intermediate = self.linear2(intermediate)
        intermediate = self.tanh(intermediate)
        intermediate = self.linear3(intermediate)
        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
        return intermediate


class Quantum_Generator(nn.Module):
    """Generatore quantistico
    Args:
        Nq (int): numero di qubit
        qlayers (List[int]): numero di ripetizioni dei layer addestrabili del cirucito quantistico
        device: QPU o simulatore quantistico (il simulatore quantistico di pennylane è stato messo di default)
        shots: numero di shot del cirucito quantistico per valutare i valori di aspettazione
    """

    def __init__(self, Nq, qlayers=2, device="default.qubit", shots = 1):
        super(Quantum_Generator, self).__init__()
        self.number_qubits = Nq
        self.shots = shots
        self.qlayers = qlayers
        self.dev = qml.device(device, wires=self.number_qubits, shots = self.shots)

        """
        Il generatore quantistico è qnode cioè un circuito quantistico con un output
        di tipo pytorch.tensor (perché nel decoratore è fissato interface="torch").
        Il qnode è composto dai seguenti circuiti concatenati:
        Input layer
        Binary Embedding:
            Un embedding binario corrisponde a prepare i qubit in uno degli stati della 
            base computazionale standard (|x>, x in {0,1}^n), questo tipo di embedding è 
            quindi costituito da soli gate X.   
        Leanable layers
        BasicEntanglerLayers: 
            Un quantum layer di questo tipo corrisponde ad un cirucito dove vengono 
            applicate per ogni qubit rotazioni lungo un asse della sfera di Bloch a picere 
            (scegliendo l'asse Y uno stato dei qubit solo con ampiezze reali) intervallate 
            da mappe di entanglement (circolare in questo caso).

        --| Ry |----@--------X---| Ry |--
                    |        |
        --| Ry |---X----@----|---| Ry |--
                        |    |
        --| Ry |--------X----@---| Ry |--

        Nel decoratore viene specificato il metodo di calcolo del gradiente.
        Il metodo parameter-shift è adatto anche a run su QPU oltre che su simulatore.
        Si basa sulla regola del parameter-shift: https://arxiv.org/pdf/1803.00745.pdf

        Gli output del qnode sono valore di aspettazione di matrici hermitiane (per il generatore mettiamo 
        Z di pauli per ogni qubit)

        """

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def qnode(inputs, weights):
            for i in range(self.number_qubits): 
                if inputs[i] > 0.5:
                    qml.X(wires=i)
            qml.templates.BasicEntanglerLayers(weights=weights, wires=range(self.number_qubits), rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.number_qubits)]

        self.weight_shapes = {"weights": [self.qlayers, self.number_qubits]}
        self.model = qml.qnn.TorchLayer(qnode, self.weight_shapes)

    def forward(self, input_tensor):
        """ Forward pass: Mappa dal latent space (vettore noise) 
        al sample space (immagine da generare) """
        intermediate = self.model(input_tensor)
        return intermediate