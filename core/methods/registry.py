from core.methods.gap.base import GAP
from core.methods.gap.edge import EdgeLevelGAP
from core.methods.gap.node import NodeLevelGAP
from core.methods.progap.base import ProGAP
from core.methods.progap.edge import EdgeLevelProGAP
from core.methods.progap.node import NodeLevelProGAP


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
