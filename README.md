# marimba

Marimba is a library for developing and training deep neural networks for the classification of astronomical time series data. 

Currently implemented architectures include convolutional neural networks and residual neural networks, where models can be specified in YAML files and training can be started using the command-line utility `marimba train`. Custom extensions to the architectures can be developed by the user by means of marimba's functional API. A variety of data processing techniques are available, including regularization with nonparametric (Gaussian Process) fitting and normalization.

This framework has been developed as part of a research project for Part III of the Mathematical Tripos at the University of Cambridge supervised by Dr. K. S. Mandel, from the Department of Pure Mathematics and Mathematical Statistics.
