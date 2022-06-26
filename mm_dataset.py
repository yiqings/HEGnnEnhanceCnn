import os
import scipy.io
from typing import Optional, Any
import numpy as np
from torch_geometric.data import Data,Dataset
import torch_geometric.transforms
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms

class MMGraphDataset(Dataset):
    """ Persistent Image dataset"""
    
    def __init__(
        self, 
        graph_path: str,
        img_path: str,
        verbosity: Optional[bool] = False,
        gnn_transform: Any = None,
        img_transform: Any = None,
        train_mode = True,
    ):
        super().__init__()
        
        self.graph_path = graph_path
        self.img_path = img_path
        
        # self.gnn_transform = gnn_transform
        if gnn_transform or not train_mode:
            self.gnn_transform = gnn_transform
        elif train_mode:
            self.gnn_transform = torch_geometric.transforms.Compose([
                    torch_geometric.transforms.RandomTranslate(5),
                    # torch_geometric.transforms.RandomFlip(0),
                ])
        
        
        if img_transform:
            self.img_transform = img_transform
        else:
            if train_mode:
                self.img_transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            else: # test mode
                self.img_transform=transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        
        for _file in os.listdir(graph_path):
            if("weighted_edge_index" in _file):
                # load edge_index
                self.edge_index = scipy.io.loadmat(os.path.join(self.graph_path, _file))['edge_index'][0]
            elif("weighted_edge_attr" in _file):
                # load edge_attr
                self.edge_attr = scipy.io.loadmat(os.path.join(self.graph_path, _file))['edge_attr'][0]
            elif("weighted_feature" in _file):
                # load feature
                feature = scipy.io.loadmat(os.path.join(self.graph_path, _file))
                self.feature = (feature['feature'][0]).reshape(-1)
            elif("weighted_label" in _file):
                # load label
                self.label = scipy.io.loadmat(os.path.join(self.graph_path, _file))['label'][0]
            elif("weighted_pid.mat" in _file):
                # load label_pid
                self.pid = scipy.io.loadmat(os.path.join(self.graph_path, _file))['pid'][0]
            elif("weighted_pid_name.mat" in _file):
                # load pid_name
                pid_name = scipy.io.loadmat(os.path.join(self.graph_path, _file))['pid_name']
            elif verbosity:
                    print('Not identified file path: {}'.format(_file))
    
        self.imgs = []
        self.labels = []
        
        
        img_dir, label_dir = self.get_img_dataset_list(self.img_path)
        for name in pid_name:
            name = name.rstrip()
            self.imgs.append(img_dir[name])
            self.labels.append(label_dir[name])
        
        assert len(self.imgs) == len(self.labels), 'Unmatched number of images {}, and labels {}'.format(len(self.imgs),len(self.labels))
        assert len(self.imgs) == len(self.feature), 'Unmatched number of images {}, and features {}'.format(len(self.imgs),len(self.feature))
        assert len(self.imgs) == len(self.edge_index), 'Unmatched number of images {}, and edge_indexs {}'.format(len(self.imgs),len(self.edge_index))
        assert len(self.imgs) == len(self.edge_attr), 'Unmatched number of images {}, and edge_attrs {}'.format(len(self.imgs),len(self.edge_attr))
        
        self.loader = default_loader
        
    def get_img_dataset_list(self, img_path: str):
        img_dir = {}
        label_dir = {}
        for category in os.listdir(img_path):
            for file in os.listdir(os.path.join(img_path,category)):
                key = file.split('.')[0]
                img_dir[key] = os.path.join(img_path,category,file)
                label_dir[key] = int(category)
        
        return img_dir, label_dir
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        
        edge_index = np.array(self.edge_index[index][:,0:2],dtype=np.int32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        edge_attr = np.array(self.edge_attr[index][:,0:1],dtype=np.int32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        feature = torch.tensor(self.feature[index], dtype=torch.float)
        label = torch.tensor([self.label[index]],dtype=torch.long)
        
        # pid = torch.tensor([self.pid[index]],dtype=torch.long)
        pid = torch.tensor([1],dtype=torch.long) ### temporial
        
        # put edge, feature, label together to form graph information in "Data" format
        graph = Data(x = feature, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=label, pid=pid)
        
        img_path = self.imgs[index]
        label = self.labels[index]
        
        _img = self.loader(img_path)
        img = self.img_transform(_img)
        return graph, img, label
    
    @property
    def num_features(self):
        data_tmp = self.__getitem__(0)
        return data_tmp[0].x.shape[1]
        # return data_tmp['x'].shape[1]
    
    @property
    def num_samples(self):
        return len(self.imgs)
    
    @property
    def num_samples_per_class(self):
        result = {}
        for i in self.labels:
            if i in result.keys():
                result[i] += 1 
            else:
                result[i] = 1
        return result
    

class MMEvalGraphDataset(MMGraphDataset):
    """ Persistent Image dataset"""
    
    def __init__(
        self, 
        graph_path: str,
        img_path: str,
        verbosity: Optional[bool] = False,
        gnn_transform: Any = None,
        img_transform: Any = None,
        train_mode = True,
    ):
        MMGraphDataset.__init__(
            self,
            graph_path,
            img_path,
            verbosity,
            gnn_transform,
            img_transform,
            train_mode,
        )

    def __getitem__(self, index):
        
        edge_index = np.array(self.edge_index[index][:,0:2],dtype=np.int32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        edge_attr = np.array(self.edge_attr[index][:,0:1],dtype=np.int32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        feature = torch.tensor(self.feature[index], dtype=torch.float)
        label = torch.tensor([self.label[index]],dtype=torch.long)
        
        # pid = torch.tensor([self.pid[index]],dtype=torch.long)
        pid = torch.tensor([1],dtype=torch.long) ### temporial
        
        # put edge, feature, label together to form graph information in "Data" format
        graph = Data(x = feature, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=label, pid=pid)
        
        img_path = self.imgs[index]
        label = self.labels[index]
        
        _img = self.loader(img_path)
        img = self.img_transform(_img)
        return graph, img, label, img_path
    