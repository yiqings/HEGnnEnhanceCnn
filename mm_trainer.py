from mm_dataset import MMGraphDataset
from mm_model import create_model
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf
import os
import torch 
from torch.optim import AdamW
from torch.optim import lr_scheduler 
from typing import Optional
from utils import (
    AverageMeter,
    AUCRecorder,
    accuracy,
)   
import time 
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random 

from const import (
    GRAPH,
    IMAGE,
    LABEL,
)

from shutil import copyfile

class MMTrainer:
    def __init__(
        self,
        config_path:str,
    ):
        config = OmegaConf.load(config_path)
        
        if config.environment.seed is not None:
            torch.manual_seed(config.environment.seed)
            np.random.seed(config.environment.seed)
            random.seed(config.environment.seed)
            
        self.device = torch.device('cuda:{}'.format(config.environment.gpu_id) if torch.cuda.is_available() else 'cpu')
        
        trainset = MMGraphDataset(
            graph_path = config.data.train.graph_path,
            img_path = config.data.train.img_path,
            train_mode = True, # turn on the augmentation
        )
        
        train_loader = DataLoader(
            dataset = trainset,
            batch_size = config.data.batch_size,
            shuffle = True,
            num_workers = config.data.num_workers,
        )
        
        testset = MMGraphDataset(
            graph_path = config.data.test.graph_path,
            img_path = config.data.test.img_path,
            train_mode = False, # turn of the augmentation for training data
        )
        
        test_loader = DataLoader(
            dataset = testset,
            batch_size = config.data.batch_size,
            shuffle = False,
            num_workers = config.data.num_workers,
        )
        
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        
        print('Complete loadding data.')
        
        os.makedirs(config.output_path.base_path,exist_ok=True)
        
        if not config.output_path.tuning:
            file_name = str(len(os.listdir(config.output_path.base_path))+1) + '_' + config.output_path.exp_file
        else:
            file_name = config.output_path.exp_file
        
        self.output_path = os.path.join(config.output_path.base_path,file_name) 
        exist_ok = config.output_path.exist_ok if config.output_path.exist_ok else True
        
        os.makedirs(self.output_path, exist_ok=exist_ok)
        os.makedirs(os.path.join(self.output_path,'weights','acc') , exist_ok=exist_ok)
        os.makedirs(os.path.join(self.output_path,'weights','auc') , exist_ok=exist_ok)
        os.makedirs(os.path.join(self.output_path,'figures') , exist_ok=exist_ok)
        
        
        self.logging = open(os.path.join(self.output_path,'logging.txt'), 'w')
        copyfile(config_path,os.path.join(self.output_path,'config.yaml'))
        
        print('Complete creating export files.')
        
        
        assert trainset.num_features == testset.num_features
        model = create_model(
            config = config.models,
            num_classes = config.data.num_classes,
            in_features = trainset.num_features)
        
        self.model = model.to(self.device)
        
        
        print('Complete creating model.')
        
        self.optimizer = AdamW(
            params = self.model.parameters(), 
            lr = config.optimization.learning_rate, 
            weight_decay = config.optimization.weight_decay
        ) 
        
        
        if config.optimization.scheduler.lower() == 'epoential':
            self.scheduler = lr_scheduler.ExponentialLR(
                optimizer = self.optimizer, 
                gamma = config.optimization.gamma
            )
        elif config.optimization.scheduler.lower() == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer = self.optimizer, 
                T_max = 10,
                eta_min = config.optimization.min_learning_rate,
            )
        elif config.optimization.scheduler.lower() == 'constant':
            self.scheduler = lr_scheduler.ConstantLR(
                optimizer = self.optimizer, 
            )
        else:
            raise ValueError("Unkown scheduler {}".format(config.optimization.scheduler.lower()))
        self.epochs = config.optimization.epochs
        self.patience = config.optimization.patience
        
    def train(
        self,
        verbosity: Optional[bool] = True,
    ):
        best_test_acc = 0.0
        best_test_auc = 0.0
        best_epoch = 0
        
        time_start=time.time()
        
        msg = 'Total training epochs : {}\n'.format(self.epochs) 
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
            
        for epoch in range(1,self.epochs+1):
            train_loss, train_acc, train_auc = self._train_one_epoch()
            test_loss, test_acc, test_auc, _test_auc_recorder = self._test_per_epoch(model=self.model)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc 
                best_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'acc' , 'best_model.pth'))
                
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_auc_epoch = epoch
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'auc' , 'model_epoch{}.pth'.format(epoch)))
                torch.save(self.model.state_dict(), os.path.join(self.output_path, 'weights' ,'auc' , 'best_model.pth'))
                
                _test_auc_recorder.draw_roc(
                    path = os.path.join(self.output_path,'figures','epoch_{}_test_roc.png'.format(epoch))
                )
            
            msg = 'Epoch {:03d} ##################   \
                \n \tTrain loss: {:.5f},   Train acc: {:.3f}%,    Train auc: {:.4f};\
                \n \tTest loss: {:.5f},   Test acc: {:.3f}%,   Test auc: {:.4f};  \
                \n \tBest test acc: {:.3f},    Best test auc: {:.4f}\n\n'.format(
                    epoch, train_loss, train_acc, train_auc,
                    test_loss, test_acc, test_auc, 
                    best_test_acc, best_test_auc)  
                
            if verbosity:
                print(msg)
            self.logging.write(msg)
            self.logging.flush()   
            
            if (epoch - best_epoch) > self.patience:
                break
        
        msg = "Best test acc:{:.3f}% @ epoch {} \n".format(best_test_acc,best_epoch)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
        
        msg = "Best test auc:{:.4f} @ epoch {} \n".format(best_test_auc,best_auc_epoch)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 
        
        
        time_end=time.time()    
        msg= "run time: {:.1f}s, {:.2f}h\n".format(time_end-time_start,(time_end-time_start)/3600)
        if verbosity:
            print(msg)
        self.logging.write(msg)
        self.logging.flush() 

    
    def _train_one_epoch(self):  
        _train_loss_recorder = AverageMeter()
        _train_acc_recorder = AverageMeter()
        _train_auc_recorder = AUCRecorder()
        
        self.model.train()
        
        for data in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            
            graph, img, label = data
            
            graph = graph.to(self.device)
            img = img.to(self.device)
            label = label.to(self.device)
            
            data = { 
                GRAPH: graph,
                IMAGE: img,
                LABEL: label,
            }
            
            
            out = self.model(data)
            loss = F.cross_entropy(out, data[LABEL])
            
            loss.backward() 
            self.optimizer.step()
                
            acc = accuracy(out, data[LABEL])[0]
            _train_loss_recorder.update(loss.item(), out.size(0))
            _train_acc_recorder.update(acc.item(), out.size(0))
            _train_auc_recorder.update(prediction=out[:,1],target=data[LABEL])

        self.scheduler.step()
        
        train_loss = _train_loss_recorder.avg 
        train_acc = _train_acc_recorder.avg 
        train_auc = _train_auc_recorder.auc
        
        return train_loss, train_acc, train_auc
    
    
    def _test_per_epoch(self, model):
        _test_loss_recorder = AverageMeter()
        _test_acc_recorder = AverageMeter()
        _test_auc_recorder = AUCRecorder()
        
        with torch.no_grad():
            model.eval()
            
            for data in tqdm(self.test_loader):
                
                graph, img, label = data
                
                graph = graph.to(self.device)
                img = img.to(self.device)
                label = label.to(self.device)
                
                data = { 
                    GRAPH: graph,
                    IMAGE: img,
                    LABEL: label,
                }
            
                out  = model(data)
                

                loss = F.cross_entropy(out, data[LABEL])
                
                acc = accuracy(out, data[LABEL])[0]

                _test_loss_recorder.update(loss.item(), out.size(0))
                _test_acc_recorder.update(acc.item(), out.size(0))
                _test_auc_recorder.update(prediction=out[:,1],target=data[LABEL])
        
        test_loss = _test_loss_recorder.avg 
        test_acc = _test_acc_recorder.avg 
        test_auc = _test_auc_recorder.auc
        
        
        return test_loss, test_acc, test_auc, _test_auc_recorder
    
    

