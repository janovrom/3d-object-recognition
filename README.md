<h1 align="center">ON IMPROVING 3D U-NET ARCHITECTURE</h1>
<p align="center">
Roman Janovský, David Sedláček and Jiří Žára 
<br/>
Faculty of Electrical Engineering, Czech Technical University in Prague, Technicka 2, Praha 6, Czechia
<br/>
roman.janovsky9@seznam.cz,{david.sedlacek, zara}@fel.cvut.cz
</p>

#### Keywords
Point cloud, Segmentation, Neural network, U-net, Voxel grid

#### Abstract
This paper presents a review of various techniques for improving the performance of neural networks on segmentation task using 3D convolutions and voxel grids – we provide comparison of network with and without max pooling, weighting, masking out the segmentation results, and oversampling results for imbalanced training dataset. We also present changes to 3D U-net architecture that give better results than the standard implementation. Although there are many out-performing architectures using different data input, we show, that although the voxel grids that serve as an input to the 3D U-net, have limits to what they can express, they do not reach their full potential.

![Poster presented on ICSOFT 2019]("Media/network_architecture.jpg")
