# RIR-Generator

The image method, proposed by Allen and Berkley in 1979 [1], is probably one of the most frequently used methods in the acoustic signal processing community to create synthetic room impulse responses. 
A mex-function, which can be used in MATLAB, was developed to generate multi-channel room impulse responses using the image method. 
This function enables the user to control the reflection order, room dimension, and microphone directivity. 

This repository includes a tutorial, MATLAB examples, and the source code of the mex-function.
A similar interface is available as a Python module and a C++ shared library.

More information can be found [here](https://www.audiolabs-erlangen.de/fau/professor/habets/software/rir-generator).

1. J.B. Allen and D.A. Berkley, "Image method for efficiently simulating small-room acoustics," Journal Acoustic Society of America, 65(4), April 1979, p 943.

The code in this folder is based on the following repositories
https://github.com/ehabets/RIR-Generator original
https://github.com/Marvin182/rir-generator python fork
https://github.com/boeddeker/rir-generator fork of python fork

