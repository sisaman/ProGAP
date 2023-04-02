from core.methods.node.base import NodeClassification
from core.methods.node.gap.base import GAP
from core.methods.node.gap.edge import EdgeLevelGAP
from core.methods.node.gap.node import NodeLevelGAP
from core.methods.node.progap.base import ProGAP
from core.methods.node.progap.edge import EdgeLevelProGAP
from core.methods.node.progap.node import NodeLevelProGAP


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
}
