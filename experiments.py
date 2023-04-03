from itertools import product
import yaml
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from core import console
from core.jobutils.registry import WandBJobRegistry
from core.jobutils.scheduler import JobScheduler, cluster_resolver
from rich.progress import Progress


def create_train_commands(registry: WandBJobRegistry) -> list[str]:
    # ### Hyper-parameters
    datasets = [
        'facebook', 'reddit', 'amazon', 'facebook-100', 'wenet'
    ]
    batch_size = {'facebook': 256, 'reddit': 2048, 'amazon': 4096, 'facebook-100': 4096, 'wenet': 1024}
    max_degree = {'facebook': 100, 'reddit': 400, 'amazon': 50, 'facebook-100': 100, 'wenet': 400}

    methods = ['progap', 'gap']
    levels = ['none', 'edge', 'node']
    hparams = {(dataset, method, level): {} for dataset, method, level in product(datasets, methods, levels)}

    for dataset in datasets:
        # Default for all methods
        for method, level in product(methods, levels):
            hparams[dataset, method, level]['hidden_dim'] = 16
            hparams[dataset, method, level]['activation'] = 'selu'
            hparams[dataset, method, level]['optimizer'] = 'adam'
            hparams[dataset, method, level]['learning_rate'] = [0.01, 0.05]
            hparams[dataset, method, level]['repeats'] = 10
            hparams[dataset, method, level]['epochs'] = 100
            hparams[dataset, method, level]['batch_size'] = 'full'

        # For ProGAP methods
        for level in levels:
            hparams[dataset, 'progap', level]['base_layers'] = [1, 2]
            hparams[dataset, 'progap', level]['head_layers'] = 1
            hparams[dataset, 'progap', level]['jk'] = 'cat'
            hparams[dataset, 'progap', level]['depth'] = [1, 2, 3, 4, 5]
            hparams[dataset, 'progap', level]['layerwise'] = False
        
        # For GAP methods
        for level in levels:
            hparams[dataset, 'gap', level]['encoder_layers'] = 2
            hparams[dataset, 'gap', level]['base_layers'] = 1
            hparams[dataset, 'gap', level]['head_layers'] = 1
            hparams[dataset, 'gap', level]['combine'] = 'cat'
            hparams[dataset, 'gap', level]['hops'] = [1, 2, 3, 4, 5]
        
        # For node-level methods
        for method in methods:
            hparams[dataset, method, 'node']['max_degree'] = max_degree[dataset]
            hparams[dataset, method, 'node']['max_grad_norm'] = 1.0
            hparams[dataset, method, 'node']['epochs'] = [5, 10]
            hparams[dataset, method, 'node']['batch_size'] = batch_size[dataset]

    progress = Progress(
        *Progress.get_default_columns(),
        "[cyan]{task.fields[registered]}[/cyan] jobs registered",
        console=console,
    )
    task = progress.add_task('generating jobs', total=None, registered=0)

    with progress:
        # ### Accuracy/Privacy Trade-off
        for dataset in datasets:
            for method in methods:
                for level in levels:
                    # copy to avoid overwriting
                    params = {**hparams[dataset, method, level]}
                    
                    if level == 'node':
                        params['epsilon'] = [2, 4, 8, 16, 32]
                    elif level == 'edge':
                        params['epsilon'] = [0.25, 0.5, 1, 2, 4]
                        
                    registry.register(
                        'train.py',
                        method, 
                        level,
                        dataset=dataset,
                        **params, 
                    )

                    progress.update(task, registered=len(registry.job_list))

        # ### Convergence
        for dataset in datasets:
            for level in levels:
                if level == 'none': continue

                # copy to avoid overwriting
                params = {**hparams[dataset, 'progap', level]}
                params['repeats'] = 1
                params['depth'] = 5

                if level == 'node':
                    params['epsilon'] = 8
                    params['epochs'] = 10
                elif level == 'edge':
                    params['epsilon'] = 1
                    params['epochs'] = 100
                
                params['log_all'] = True
                registry.register(
                    'train.py',
                    method, 
                    level,
                    dataset=dataset,
                    **params,
                )

                progress.update(task, registered=len(registry.job_list))

        # ### Progressive vs. Layer-wise
        for dataset in datasets:
            for level in levels:
                # copy to avoid overwriting
                params = {**hparams[dataset, 'progap', level]}
                
                if level == 'node':
                    params['epsilon'] = 8
                elif level == 'edge':
                    params['epsilon'] = 1
                
                params['layerwise'] = True
                registry.register(
                    'train.py',
                    method, 
                    level,
                    dataset=dataset,
                    **params,
                )

                params['layerwise'] = False
                registry.register(
                    'train.py',
                    method, 
                    level,
                    dataset=dataset,
                    **params,
                )

                progress.update(task, registered=len(registry.job_list))

    return registry.job_list


def generate(path: str):
    """Generate experiment job file.

    Args:
        path (str): Path to store job file.
    """
    with open('config/wandb.yaml') as f:
        wandb_config = yaml.safe_load(f)

    registry = WandBJobRegistry(**wandb_config)
    registry.pull()
    create_train_commands(registry)
    registry.save(path=path)
    console.info(f'job file saved to [bold blue]{path}[/bold blue]')


def run(job_file: str, scheduler_name: str) -> None:
    """Run jobs in parallel using a distributed job scheduler.

    Args:
        job_file (str): Path to the job file.
        scheduler_name (str): Name of the scheduler to use.
    """

    with open('config/dask.yaml') as f:
        config = yaml.safe_load(f)

    scheduler = JobScheduler(
        job_file=job_file, 
        scheduler=scheduler_name, 
        config=config
    )
    
    try:
        scheduler.submit()
    except KeyboardInterrupt:
        console.warning('Graceful shutdown')



def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generate', action='store_true', help='Generate jobs')
    parser.add_argument('--run', action='store_true', help='Run jobs')
    parser.add_argument('--path', type=str, default='jobs/experiments.sh', help='Path to the job file')
    parser.add_argument('--scheduler', type=str, default='sge', help='Job scheduler to use', 
                        choices=cluster_resolver.options)
    args = parser.parse_args()

    if args.generate:
        generate(args.path)
    if args.run:
        run(job_file=args.path, scheduler_name=args.scheduler)
    
    if not args.generate and not args.run:
        parser.error('Please specify either --generate or --run')


if __name__ == '__main__':
    main()
