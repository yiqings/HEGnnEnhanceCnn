# Graph Neural Networks and Convolutional Neural Networks Fusion Model for Histology

The implementations for the paper "[How GNNs Facilitate CNNs in Mining Geometric Information from Large-Scale Medical Images](https://arxiv.org/abs/2206.07599)".


### 1. Abstract
Gigapixel medical images provide massive data, both morphological textures and spatial information, to be mined. Due to the large data scale in histology, deep learning methods play an increasingly significant role as feature extractors. Existing solutions heavily rely on convolutional neural networks (CNNs) for global pixel-level analysis, leaving the underlying local geometric structure such as the interaction between cells in the tumor microenvironment unexplored. The topological structure in medical images, as proven to be closely related to tumor evolution, can be well characterized by graphs. To obtain a more comprehensive representation for downstream oncology tasks, we propose a fusion framework for enhancing the global image-level representation captured by CNNs with the geometry of cell-level spatial information learned by graph neural networks (GNN). The fusion layer optimizes an integration between collaborative features of global images and cell graphs. Two fusion strategies have been developed: one with MLP which is simple but turns out efficient through fine-tuning, and the other with Transformer gains a champion in fusing multiple networks. We evaluate our fusion strategies on histology datasets curated from large patient cohorts of colorectal and gastric cancers for three biomarker prediction tasks. Both two models outperform plain CNNs or GNNs, reaching a consistent AUC improvement of more than 5% on various network backbones. The experimental results yield the necessity for combining image-level morphological features with cell spatial relations in medical image analysis. 


### 2. Dataset
The images for MSI predictions are avaiable at https://zenodo.org/record/2530835#.YrLH-S-KFtQ. It comprises of 
- STAD_TRAIN_MSS: training images or gastric cancer patients with MSS tumor.
- STAD_TRAIN_MSIMUT: training images or gastric cancer patients with MSI tumor.
- STAD_TEST_MSS: test images or gastric cancer patients with MSS tumor.
- STAD_TEST_MSIMUT: test images or gastric cancer patients with MSS tumor.
- CRC_TRAIN_MSS: training images or colorectal cancer patients with MSS tumor.
- CRC_TRAIN_MSIMUT: training images or colorectal cancer patients with MSI tumor.
- CRC_TEST_MSS: test images or colorectal cancer patients with MSS tumor.
- CRC_TEST_MSIMUT: test images or colorectal cancer patients with MSS tumor.

The assoicated graph data data is avaiable at https://zenodo.org/record/6683652#.YrLjLC-KFtQ. It comprises of
- stad_msi_train: The cell-graph extracted from STAD_TRAIN_MSS and STAD_TRAIN_MSIMUT.
- stad_msi_test: The cell-graph extracted from STAD_TEST_MSS and STAD_TEST_MSIMUT.
- crc_msi_train: The cell-graph extracted from CRC_TRAIN_MSS and CRC_TRAIN_MSIMUT.
- crc_msi_test: The cell-graph extracted from CRC_TEST_MSS and CRC_TEST_MSIMUT. 

### 3. Code Organization

- [`mm_model.py`](mm_model.py): model construction.
- [`mm_trainer.py`](mm_trainer.py): the training codes.
- [`mm_evaluater.py`](mm_evaluater.py): the evaluation codes for both patch and WSI levels.
- [`mm_dataset.py`](mm_dataset.py): dataset loader.
- [`main.py`](main.py): main functions.

---
If you find our paper, code or graph data helpful in your research. Please consider citing our paper:
```
@article{shen2022gnns,
  title={How GNNs Facilitate CNNs in Mining Geometric Information from Large-Scale Medical Images},
  author={Shen, Yiqing and Zhou, Bingxin and Xiong, Xinye and Gao, Ruitian and Wang, Yu Guang},
  journal={arXiv preprint arXiv:2206.07599},
  year={2022}
}
```



