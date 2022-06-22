# Graph Neural Networks and Convolutional Neural Networks Fusion Model for Histology


### 1. Abstract
Gigapixel medical images provide massive data, both morphological textures and spatial information, to be mined. Due to the large data scale in histology, deep learning methods play an increasingly significant role as feature extractors. Existing solutions heavily rely on convolutional neural networks (CNNs) for global pixel-level analysis, leaving the underlying local geometric structure such as the interaction between cells in the tumor microenvironment unexplored. The topological structure in medical images, as proven to be closely related to tumor evolution, can be well characterized by graphs. To obtain a more comprehensive representation for downstream oncology tasks, we propose a fusion framework for enhancing the global image-level representation captured by CNNs with the geometry of cell-level spatial information learned by graph neural networks (GNN). The fusion layer optimizes an integration between collaborative features of global images and cell graphs. Two fusion strategies have been developed: one with MLP which is simple but turns out efficient through fine-tuning, and the other with Transformer gains a champion in fusing multiple networks. We evaluate our fusion strategies on histology datasets curated from large patient cohorts of colorectal and gastric cancers for three biomarker prediction tasks. Both two models outperform plain CNNs or GNNs, reaching a consistent AUC improvement of more than 5% on various network backbones. The experimental results yield the necessity for combining image-level morphological features with cell spatial relations in medical image analysis. 


### 2. Dataset
The images are avaiable at https://zenodo.org/record/2530835#.YrLH-S-KFtQ.

The assoicated graph data data is avaiable at https://zenodo.org/record/6683652#.YrLjLC-KFtQ.
