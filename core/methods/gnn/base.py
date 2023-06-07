from typing import Annotated
from class_resolver.contrib.torch import activation_resolver
from core.args.utils import ArgInfo
from core.methods.base import NodeClassification
from core.modules.base import TrainableModule
from core.modules.gnn import GNNNodeClassifier


class StandardGNN (NodeClassification):
    """Non-private GNN method"""
    
    def __init__(self,
                 num_classes,
                 conv:                  Annotated[str,   ArgInfo(help='type of convolution layer', choices=['sage', 'gcn', 'gin', 'gat'])] = 'sage',
                 hidden_dim:            Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 base_layers:           Annotated[int,   ArgInfo(help='number of base MLP layers')] = 1,
                 mp_layers:             Annotated[int,   ArgInfo(help='number of message passing layers')] = 2,
                 head_layers:           Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 activation:            Annotated[str,   ArgInfo(help='type of activation function', choices=['relu', 'selu', 'tanh'])] = 'selu',
                 dropout:               Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:            Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 jk:                    Annotated[str,   ArgInfo(help='the jumping knowledge mode.', choices=["last", "cat", "max", "lstm"])] = 'cat',
                 # sage args
                 sage_aggr:             Annotated[str,   ArgInfo(help='SAGE: type of aggregation function', choices=['mean', 'sum', 'max', 'lstm'])] = 'mean',
                 sage_root_weight:      Annotated[bool,  ArgInfo(help='SAGE: if true, will add transformed root node features to the output')] = True,
                 sage_normalize:        Annotated[bool,  ArgInfo(help='SAGE: if true, output features will be L2-normalized')] = True,
                 # gcn args
                 gcn_improved:          Annotated[bool,  ArgInfo(help='GCN: if true, will use improved version')] = False,
                 gcn_add_self_loops:    Annotated[bool,  ArgInfo(help='GCN: if true, will add self loops to the adjacency matrix')] = True,
                 gcn_cached:            Annotated[bool,  ArgInfo(help='GCN: if true, will cache the normalized adjacency matrix')] = True,
                 gcn_normalize:         Annotated[bool,  ArgInfo(help='GCN: whether to add self-loops and compute symmetric normalization coefficients on the fly')] = True,
                 # gat args
                 gat_heads:             Annotated[int,   ArgInfo(help='GAT: number of attention heads')] = 1,
                 gat_concat:            Annotated[bool,  ArgInfo(help='GAT: if true, will concatenate multi-head attentions; otherwise, will average them')] = True,
                 gat_negative_slope:    Annotated[float, ArgInfo(help='GAT: negative slope of the leaky relu activation')] = 0.2,
                 gat_add_self_loops:    Annotated[bool,  ArgInfo(help='GAT: if true, will add self loops to the adjacency matrix')] = True,
                 # optimizer args
                 optimizer:             Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:         Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:          Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 **kwargs:              Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'
        
        self.num_classes = num_classes
        self.conv = conv
        self.hidden_dim = hidden_dim
        self.base_layers = base_layers
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.activation_fn = activation_resolver.make(activation)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.jk = jk
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if self.conv == 'sage':
            self.conv_kwargs = {
                'aggr': sage_aggr,
                'root_weight': sage_root_weight,
                'normalize': sage_normalize,
            }
        elif self.conv == 'gcn':
            self.conv_kwargs = {
                'improved': gcn_improved,
                'add_self_loops': gcn_add_self_loops,
                'cached': gcn_cached,
                'normalize': gcn_normalize,
            }
        elif self.conv == 'gat':
            self.conv_kwargs = {
                'heads': gat_heads,
                'concat': gat_concat,
                'negative_slope': gat_negative_slope,
                'add_self_loops': gat_add_self_loops,
            }

        super().__init__(num_classes, **kwargs)

    def configure_classifier(self) -> TrainableModule:
        return GNNNodeClassifier(
            hidden_dim=self.hidden_dim, 
            num_classes=self.num_classes, 
            base_layers=self.base_layers,
            mp_layers=self.mp_layers, 
            head_layers=self.head_layers, 
            conv=self.conv,
            conv_kwargs=self.conv_kwargs,
            activation_fn=self.activation_fn, 
            dropout=self.dropout, 
            batch_norm=self.batch_norm,
            jk=self.jk,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )