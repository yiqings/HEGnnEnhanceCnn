import torch
from torch import nn
from typing import List, Optional, Tuple
from models import MLP, GCNResNet
from utils import init_weights
from const import(
    GRAPH,
    IMAGE,
    LABEL,
    LOGITS, 
    FEATURES,
    GNN_MODEL,
    TIMM_MODEL,
    FUSION_MLP,
    FUSION_TRANSFORMER,
    GNN_TRANSFORMER,
    GNN_RESNET,
)

from models import gnn_model_dict,fusion_dict
import timm 
import functools
from ft_transformer import FT_Transformer

class TIMM(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
            num_classes: Optional[int] = 0,
            pretrained: Optional[bool] = True,
    ):
        super(TIMM, self).__init__()
        self.prefix = prefix 
        
        # self.data_key = f"{prefix}_{IMAGE}"
        # self.label_key = f"{prefix}_{LABEL}"
        
        self.data_key = f"{IMAGE}"
        self.label_key = f"{LABEL}"
        
        self.num_classes = num_classes
        
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        
        self.out_features = self.model.num_features
        
        self.head = nn.Linear(self.out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(init_weights)
        
        
     
    def forward(
        self, 
        batch
    ):
        data = batch[self.data_key]
        
        features = self.model(data)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }     
        

class GNN(nn.Module):
    def __init__(
            self,
            prefix: str,
            model_name: str,
            num_classes: int,
            in_features: int, 
            hidden_features: Optional[int] = 256, 
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        
        super(GNN, self).__init__()
        
        self.prefix = prefix 
        self.num_classes = num_classes
        
        # self.data_key = f"{prefix}_{GRAPH}"
        # self.label_key = f"{prefix}_{LABEL}"
        
        self.data_key = f"{GRAPH}"
        self.label_key = f"{LABEL}"
        
        self.model = gnn_model_dict[model_name](
            in_features = in_features,
            hidden_features = hidden_features,
            out_features = out_features,
            pooling = pooling,
            activation = activation,
        )
        
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        self.out_features = out_features
        
        
    def forward(self, batch):
        data = batch[self.data_key]
        
        features = self.model(data)
        logits = self.head(features)
            
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }



class FusionMLP(nn.Module):
    def __init__(
        self,
        prefix: str,
        models: list,
        num_classes: int,
        hidden_features: List[int],
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
    ):
    
        super().__init__()
        self.prefix = prefix
        self.model = nn.ModuleList(models)   
        
        # TODO: Add out_features to each model
        raw_in_features = [per_model.out_features for per_model in models]
        
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )
            
            in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)
        
        
        fusion_mlp = []
        for per_hidden_features in hidden_features:
            fusion_mlp.append(
                MLP(
                    in_features=in_features,
                    hidden_features=per_hidden_features,
                    out_features=per_hidden_features,
                    num_layers=1,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    normalization=normalization,
                )
            )
            in_features = per_hidden_features
            
        self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        self.head = nn.Linear(in_features, num_classes)
        # init weights
        
        self.adapter.apply(init_weights)
        self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)
        
    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []

        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_features.append(per_adapter(per_output[per_model.prefix][FEATURES]))
            

        features = self.fusion_mlp(torch.cat(multimodal_features, dim=1))
        logits = self.head(features)
        # fusion_output = {
        #     self.prefix: {
        #         LOGITS: logits,
        #         FEATURES: features,
        #     }
        # }
        
        # return fusion_output
        return logits

  

class FusionTransformer(nn.Module):

    def __init__(
            self,
            prefix: str,
            models: list,
            hidden_features: int,
            num_classes: int,
            adapt_in_features: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = nn.ModuleList(models)

        raw_in_features = [per_model.out_features for per_model in models]
        if adapt_in_features is not None:
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            self.adapter = nn.ModuleList(
                [nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features]
            )

            in_features = base_in_feat
        else:
            self.adapter = nn.ModuleList(
                [nn.Identity() for _ in range(len(raw_in_features))]
            )
            in_features = sum(raw_in_features)

        assert len(self.adapter) == len(self.model)

        self.fusion_transformer = FT_Transformer(
            d_token=in_features,
            n_blocks=1,
            attention_n_heads=4,
            attention_dropout=0.1,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_d_hidden=192,
            ffn_dropout=0.1,
            ffn_activation='ReGLU',
            ffn_normalization='LayerNorm',
            residual_dropout=0.0,
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=None,
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation='ReLU',
            head_normalization='LayerNorm',
            d_out=hidden_features,
        )

        self.head = FT_Transformer.Head(
            d_in=in_features,
            d_out=num_classes,
            bias=True,
            activation='ReLU', 
            normalization='LayerNorm',
        )
        
        # init weights
        self.adapter.apply(init_weights)
        self.head.apply(init_weights)

        self.prefix = prefix
        

    def forward(
            self,
            batch: dict,
    ):
        multimodal_features = []
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = per_model(batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature,dim=1)
            multimodal_features.append(multimodal_feature)


        multimodal_features = torch.cat(multimodal_features, dim=1)
        features = self.fusion_transformer(multimodal_features)
        logits = self.head(features)
        
        return logits



class GNNTransformer(nn.Module):
    def __init__(
        self,
        prefix: str,
        num_classes: int,  
        gnn_in_features: int, 
        model_name: Optional[str] = 'gcn',
        attn_heads: Optional[int] = 8, 
        dim_head: Optional[int] = 64, 
        emb_dropout: Optional[float] = 0.,
        hidden_features: Optional[int] = 256, 
        out_features: Optional[int] = 256,
        gnn_pooling: Optional[float] = 0.5,
        gnn_activation: Optional[str] = "gelu",
        vit_pool: Optional[str] ='cls', 
        vit_dropout: Optional[float] = 0., 
        input_image_channels: Optional[int] = 3, 
        patch_size: Optional[Tuple[int]] = (16,16), 
        image_size: Optional[Tuple[int]] = (224,224),      
    ):
        super().__init__()
        self.prefix = prefix 
        self.out_features = out_features
        
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.model = fusion_dict[model_name](
            gnn_in_features = gnn_in_features, 
            attn_heads = attn_heads, 
            dim_head = dim_head, 
            emb_dropout = emb_dropout,
            hidden_features = hidden_features, 
            out_features = out_features,
            gnn_pooling = gnn_pooling,
            gnn_activation = gnn_activation,
            vit_pool = vit_pool, 
            vit_dropout = vit_dropout, 
            input_image_channels = input_image_channels, 
            patch_size = patch_size, 
            image_size = image_size, 
        )
    
    def forward(self,batch):

        features = self.model(
            img = batch[IMAGE], 
            data = batch[GRAPH]
        ) 
        
        logits = self.head(features)
        
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }  


class GNNResNet(nn.Module):
    def __init__(
        self,
        prefix: str,
        num_classes: int,  
        gnn_in_features: int, 
        out_features: Optional[int] = 256,    
    ):
        super().__init__()
        self.prefix = prefix 
        self.out_features = out_features
        
        self.head = nn.Linear(out_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.model = GCNResNet(
            gnn_in_features=gnn_in_features,
        )
    
    def forward(self,batch):

        features = self.model(
            img = batch[IMAGE], 
            data = batch[GRAPH]
        ) 
        
        logits = self.head(features)
        
        return {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }  
        
def create_model(config, num_classes, in_features):
    models = []
    for model_name in config.names:
        model_config = getattr(config, model_name)
        if model_name.lower().startswith(GNN_MODEL):
            model = GNN(
                    prefix = model_name,
                    model_name = model_config.model_name,
                    in_features = in_features,
                    num_classes = num_classes,
                    hidden_features = model_config.hidden_features, 
                    out_features  = model_config.out_features,
                    pooling = model_config.pooling,
                    activation = model_config.activation,
            )
            
        elif model_name.lower().startswith(TIMM_MODEL):
            model = TIMM(
                    prefix = model_name,
                    model_name = model_config.model_name,
                    num_classes = num_classes,
                    pretrained = model_config.pretrained
            )
        elif model_name.lower().startswith(GNN_TRANSFORMER):
            model = GNNTransformer(
                    prefix = model_name,
                    model_name = model_config.model_name,
                    num_classes = num_classes,  
                    gnn_in_features = in_features,
                    attn_heads = model_config.attn_heads, 
                    dim_head = model_config.dim_head, 
                    emb_dropout = model_config.emb_dropout,
                    hidden_features = model_config.hidden_features, 
                    out_features = model_config.hidden_features ,
                    gnn_pooling = model_config.gnn_pooling,
                    vit_dropout = model_config.vit_dropout, 
                    gnn_activation = model_config.gnn_activation,
            )
        elif model_name.lower().startswith(GNN_RESNET):
            model = GNNResNet(
                    prefix = model_name,
                    num_classes = num_classes, 
                    gnn_in_features = in_features, 
            )
        elif model_name.lower().startswith(FUSION_MLP):
            fusion_model = functools.partial(
                    FusionMLP,
                    prefix = model_name,
                    num_classes = num_classes,
                    hidden_features = model_config.hidden_features,
                    adapt_in_features = model_config.adapt_in_features,
                    activation = model_config.activation,
                    dropout_prob = model_config.dropout_prob,
                    normalization = model_config.normalization,
            )
        elif model_name.lower().startswith(FUSION_TRANSFORMER):
            fusion_model = functools.partial(
                FusionTransformer,
                prefix = model_name,
                num_classes = num_classes,
                hidden_features = model_config.hidden_features,
                adapt_in_features = model_config.adapt_in_features,
            )
        else:
            raise ValueError(f"unknown model name: {model_name}")
        
        models.append(model)
        
    return fusion_model(models=models)