from .gnn import(
    GINTopK4,
    GINTopK2,
    GCNTopK4,
    GCNTopK2
)

from .resnet import(
   resnet32,
   resnet110,
   wide_resnet20_8,
)

from .mlp import MLP 
# from .trans_gnn import GCNViT
# from .resnet_gnn import GCNResNet

gnn_model_dict = {
    'gintopk4' : GINTopK4,
    'gintopk2' : GINTopK2,
    'gcntopk4' : GCNTopK4,
    'gcntopk2' : GCNTopK2,
}

cnn_model_dict = {
    'resnet32' : resnet32, 
    'resnet110': resnet110, 
    'wide_resnet20_8': wide_resnet20_8,
}

# fusion_dict = {
#     'gcn':GCNViT,
# }

