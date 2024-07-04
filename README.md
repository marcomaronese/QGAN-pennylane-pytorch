# QGAN-pennylane-pytorch

I have implemented a quantum GAN that reproduces elements of the
Bars and Stripes dataset. The Bars and Stripes dataset consists of
rectangular images that contain only black or white pixels. Everything is fine
image is "striped" (all pixels in the same row are of the
same color) or "bar" (all pixels in the same column
they are the same color). Completely black images are excluded
or white. The code prints the loss of both the generator and the
discriminator during learning.
As the parameters are currently set, the dataset considered
it is composed of 2x2 images (which are passed to the networks as vectors).

Code structure and workflow:
The code consists of a MAIN which

1. Load the parameters for the run from the "parameters.py" file

2. Create the discriminator and generator objects, respectively
 imported from "Generators.py" and "Discriminator.py".
 In Generators.py there is both a classic generator and a
 quantum generator. The boolean variable "isquantum" in the
 parameters.py file decides which of the two is imported.

3. Create the GAN object, calling the imported class from "GAN.py"
 (to which generator and discriminator are provided as input)

4. Import the function that generates dataset elements from "utils.py"

5. For each epoch iterate the training functions, printing the losses
 and printing an example of the vectors produced by the GAN every 20
 eras.

Quantum Implementation Details:

I implemented on pennylane, with pytorch interface,
a qGAN as described in the paper

https://www.nature.com/articles/s41534-019-0223-2

which represents the implementation on qiskit.
In the paper, qGAN is used for loading random distributions
while in our case it was applied to a generation of images.
The qGAN therefore consists of a classical NN as a discriminator
and by a quantum circuit as a generator.
A bit string (z) is sampled from a uniform distribution e
it is encoded in the state of the qubits.
A dependent unit is then applied to the qubit register
from trainable parameters.
The output of the generator is a measure (x') of the qubit register versus
to gate Z and then I get an array with values ​​1 and -1 matching
to the black and white pixels of the image.
NB: Further information on the quantum generator circuit is
present in the Generators.py file.


Comments:

With a classic generator you can notice it, even after a few hundred
eras, the phenomenon of mode collapse.
There is no evidence to suggest that a quantum generator is capable
to overcome this problem and in our tests we did not observe one
convergence of the quantum generator since the training requires a lot
longer than the classic counterpart.
This is due to the gradient calculation method it requires for each
parameter the simulation of two circuits.
I spoke to an internal IBM researcher who confirmed that
with 4 qubits the convergence is reached on the order of 1000 epochs even if
in his case the task was the reproduction of a multimodal distribution.
The choice of our task was made to have a more direct vision of
work of the quantum generator given that each measurement of the qubits corresponds
An image. In the case of the task of the article mentioned above or in the case of a
multimodal distribution we have that the measurement of qubits is converted from
binary string with decimal value creating a hierarchy between the multiple qubits
significant and less significant ones making training more complicated
(based on what we were told).
We think that with more time available to do hyperparameter tuning
it is possible to see a convergence even in the quantum case.
