from gnnprivtool.methods.gap.base import GAP
from gnnprivtool.methods.gap.edge import EdgeLevelGAP
from gnnprivtool.methods.gap.node import NodeLevelGAP
from gnnprivtool.methods.gnn.base import StandardGNN
from gnnprivtool.methods.gnn.edge import EdgeLevelGNN
from gnnprivtool.methods.gnn.node import NodeLevelGNN
from gnnprivtool.methods.lpgnet.base import LPGNet
from gnnprivtool.methods.lpgnet.edge import EdgeLevelLPGNet
from gnnprivtool.methods.mlp.edge import SimpleMLP
from gnnprivtool.methods.mlp.node import PrivateMLP
from gnnprivtool.methods.progap.base import ProGAP
from gnnprivtool.methods.progap.edge import EdgeLevelProGAP
from gnnprivtool.methods.progap.node import NodeLevelProGAP

supported_methods = {
    'progap': {
        'none': ProGAP,
        'edge': EdgeLevelProGAP,
        'node': NodeLevelProGAP,
    },
    'gap': {
        'none': GAP,
        'edge': EdgeLevelGAP,
        'node': NodeLevelGAP,
    },
    'mlp': {
        'edge': SimpleMLP,
        'node': PrivateMLP,
    },
    'gnn': {
        'none': StandardGNN,
        'edge': EdgeLevelGNN,
        'node': NodeLevelGNN,
    },
    'lpgnet': {
        'none': LPGNet,
        'edge': EdgeLevelLPGNet,
    },
}
