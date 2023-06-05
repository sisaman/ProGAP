from core.methods.gap.base import GAP
from core.methods.gap.edge import EdgeLevelGAP
from core.methods.gap.node import NodeLevelGAP
from core.methods.progap.base import ProGAP
from core.methods.progap.edge import EdgeLevelProGAP
from core.methods.progap.node import NodeLevelProGAP
from core.methods.mlp.edge import SimpleMLP
from core.methods.mlp.node import PrivateMLP


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
}
