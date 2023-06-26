from core import console, setup_console_logging
with console.status('importing modules'):
    import torch
    import numpy as np
    from rich import box
    from rich.table import Table
    from time import time
    from typing import Annotated
    from argparse import ArgumentParser
    from core import globals
    from core.datasets.loader import DatasetLoader
    from core.args.utils import print_args, create_arguments, strip_kwargs, ArgInfo
    from core.args.formatter import ArgumentDefaultsRichHelpFormatter
    from core.loggers.logger import Logger
    from core.methods.base import NodeClassification
    from core.methods.registry import supported_methods
    from core.utils import confidence_interval
    from torch_geometric import seed_everything


def run(seed:        Annotated[int,  ArgInfo(help='initial random seed')] = 12345,
        repeats:     Annotated[int,  ArgInfo(help='number of times the experiment is repeated')] = 1,
        log_trainer: Annotated[bool, ArgInfo(help='log all training steps')] = False,
        debug:       Annotated[bool, ArgInfo(help='enable global debug mode')] = False,
        **kwargs
    ):

    seed_everything(seed)

    ### setup debug mode ###
    if debug:
        console.warning('debug mode enabled')
        globals['debug'] = True
        console.log_level = console.DEBUG
        kwargs['logger'] = 'wandb'
        kwargs['log_trainer'] = True
        kwargs['project'] += '-debug'
        console.warning(f'wandb logger is active for project {kwargs["project"]}')

    ### setup logger ###
    config = {**kwargs, 'seed': seed, 'repeats': repeats}
    logger_args = strip_kwargs(Logger, kwargs)
    logger = Logger(config=config, **logger_args)
    del kwargs['logger']

    with console.status('loading dataset'):
        loader_args = strip_kwargs(DatasetLoader, kwargs)
        data = DatasetLoader(**loader_args).load(verbose=True)

    ### initiallize method ###
    num_classes = data.y.max().item() + 1
    Method = supported_methods[kwargs['method']][kwargs['level']]
    method_args = strip_kwargs(Method, kwargs)
    method_args['logger'] = logger if log_trainer else None
    method: NodeClassification = Method(num_classes=num_classes, **method_args)

    ### run experiment ###
    run_metrics = {}
    for iteration in range(repeats):
        start_time = time()
        metrics = method.run(data)
        end_time = time()
        duration = end_time - start_time
        metrics['duration'] = duration

        ### process results ###
        for metric, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

         ### print results ###
        table = Table(title=f'run {iteration + 1}: {duration:.2f} s', box=box.HORIZONTALS)
        table.add_column('metric')
        table.add_column('last', style="cyan")
        table.add_column('mean', style="cyan")
        table.add_row('test/acc', f'{run_metrics["test/acc"][-1]:.2f}', f'{np.mean(run_metrics["test/acc"]):.2f}')
        console.info(table)
        console.print()

        ### reset method's parameters for the next run ###
        method.reset()

    summary = {}
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=seed)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            summary['gpu_mem'] = gpu_mem
        logger.log_summary(summary)


def main():
    setup_console_logging()
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    method_subparser = init_parser.add_subparsers(dest='method', required=True, title='algorithm')

    for method_name, levels in supported_methods.items():
        method_parser = method_subparser.add_parser(
            name=method_name, 
            # help=method_class.__doc__, 
            formatter_class=ArgumentDefaultsRichHelpFormatter
        )

        level_subparser = method_parser.add_subparsers(dest='level', required=True, title='privacy level')

        for level_name, method_class in levels.items():
            level_parser = level_subparser.add_parser(
                name=level_name, 
                help=f'privacy level {level_name}', 
                formatter_class=ArgumentDefaultsRichHelpFormatter
            )

            # dataset args
            group_dataset = level_parser.add_argument_group('dataset arguments')
            create_arguments(DatasetLoader, group_dataset)

            # method args
            group_method = level_parser.add_argument_group('method arguments')
            create_arguments(method_class, group_method)
            
            # experiment args
            group_expr = level_parser.add_argument_group('experiment arguments')
            create_arguments(run, group_expr)
            create_arguments(Logger, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsRichHelpFormatter)
    kwargs = vars(parser.parse_args())
    print_args(kwargs, num_cols=2)

    try:
        start = time()
        run(**kwargs)
        end = time()
        console.info(f'\nTotal running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        console.warning('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


if __name__ == '__main__':
    main()
