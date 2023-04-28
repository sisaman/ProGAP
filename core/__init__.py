import rich
import logging
import warnings
from rich.traceback import install
from rich.logging import RichHandler
from core.console.console import Console

globals = {
    'debug': False,
}


# define main and error consoles
console = Console(tab_size=4)
error_console = Console(stderr=True, tab_size=4)

# setup console for tracebacks
install(console=error_console, width=error_console.width)

# set global console
rich._console = console
rich.reconfigure = lambda *args, **kwargs: None


def setup_console_logging():
    # create logger
    log_handler = RichHandler(
        console=console, 
        omit_repeated_times=True,
        log_time_format="[%X]"
    )
    logger = logging.getLogger('gap')
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    for name in logging.root.manager.loggerDict:
    # for name in ['logging.root.manager.loggerDict']:
        logger = logging.getLogger(name)
        
        if logger.handlers:
            logger.handlers.clear()
            logger.addHandler(log_handler)

        if logger.level < logging.WARNING:
            logger.setLevel(logging.WARNING)

    # setup warnings
    logging.getLogger("py.warnings").addHandler(log_handler)
    logging.getLogger("py.warnings").propagate = False
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", module="torch_geometric.utils.sparse", lineno=176)
    warnings.filterwarnings("ignore", module="scipy.optimize._optimize", lineno=2769)
    warnings.filterwarnings("ignore", module="torch.nn.modules.module", lineno=1344)