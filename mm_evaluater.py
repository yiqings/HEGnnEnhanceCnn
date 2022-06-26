from tkinter import image_names
from mm_dataset import MMGraphDataset,MMEvalGraphDataset
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
import csv 
from itertools import islice
from const import (
    GRAPH,
    IMAGE,
    LABEL,
)
from sklearn import metrics
from shutil import copyfile
import argparse

def compute_result(csv_path,output_path):
    txt_writer = open(output_path,'w')
    csv_reader = csv.reader(open(csv_path,'r')) 
    all_pred = {}
    all_label = {}
    for row in islice(csv_reader,1,None):
        name = row[0]
        if 'TCGA' in name:
            pid = name.split('-')[3] + '-' + name.split('-')[4]
        else:
            pid = name.split('_')[0] + '-' + name.split('_')[2]
    
        label = row[1]
        pred = row[5]
        
        
        if pid not in all_pred.keys():
            all_pred[pid] = [float(pred)]
            all_label[pid] = int(label)
        else:
            all_pred[pid].append(float(pred))
    
    all_pred_list = []
    all_label_list = []
    for pid in all_pred.keys():
        all_pred_list.append(
            np.mean(all_pred[pid])
        )
        all_label_list.append(
           all_label[pid]
        )
        
    
    y = np.array(all_label_list)
    pred = np.array(all_pred_list)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    acc = metrics.accuracy_score(y, pred > 0.5 )
    auc = metrics.auc(fpr, tpr) 
    
    msg = 'Acc: {} '.format(acc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
    msg = 'AUC: {} '.format(auc)
    txt_writer.write(msg + '\n')
    txt_writer.flush()
    
class MMEvaluater:
    def __init__(
        self,
        input_path:str,
        gpu_id: int,
        choice: Optional[str] = 'acc',
    ):
        config = OmegaConf.load(os.path.join(input_path,'config.yaml'))
        
        self.choice = choice
        if config.environment.seed is not None:
            torch.manual_seed(config.environment.seed)
            np.random.seed(config.environment.seed)
            random.seed(config.environment.seed)
            
        self.device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        
        testset = MMEvalGraphDataset(
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
        
        self.test_loader = test_loader 
        
       
        self.output_path = input_path
        
        model = create_model(
            config = config.models,
            num_classes = config.data.num_classes,
            in_features = testset.num_features
        )
        
        model.load_state_dict(
            torch.load(os.path.join(input_path,'weights',choice,'best_model.pth'))
        )
        
        self.model = model.to(self.device)
        print('Initalization complete.')
        
        
    def evaluate(self):
        csv_path = os.path.join(self.output_path,'prediction_{}.csv'.format(self.choice))
        txt_path = os.path.join(self.output_path,'patient_result_{}.txt'.format(self.choice))
        f = open(csv_path,'w',encoding='utf-8')
        
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            ['name','label','logits_0','logits_1','prob_0','prob_1']
        )
        

        with torch.no_grad():
            self.model.eval()
            
            for data in tqdm(self.test_loader):
                
                graph, img, label, img_path = data
                
                graph = graph.to(self.device)
                img = img.to(self.device)
                label = label.to(self.device)
                
                data = { 
                    GRAPH: graph,
                    IMAGE: img,
                    LABEL: label,
                }
            
                out  = self.model(data)
                
                out = out.cpu().numpy()
                label = label.cpu().tolist()
                img_name = [name.split('/')[-1] for name in img_path]
                
                for idx in range(len(img_name)):
                    logits = np.array(out[idx,:].tolist())
                    probs = np.exp(logits) / np.sum(np.exp(logits))
            
                    result = [
                        img_name[idx],
                        label[idx],]
                    
                    result = result + list(logits) + list(probs)
                    
                    csv_writer.writerow(result)
                
        compute_result(csv_path,txt_path)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', type=str)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()  
    
    for path in tqdm(os.listdir(args.path)):
        input_path = os.path.join(args.path,path)
        
        break_flag = False
        for name in os.listdir(input_path):
            if 'patient_result' in name:
                break_flag = True 
        
        print(input_path)
        
        if break_flag: 
            print('skip')
            continue  
         
        
        mmevalute = MMEvaluater(
            input_path=input_path,
            gpu_id=args.gpu_id,
        )
        mmevalute.evaluate()
    
