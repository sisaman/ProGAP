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

    progap_methods  = ['progap-inf', 'progap-edp', 'progap-ndp']
    # gap_methods  = ['gap-inf', 'gap-edp', 'gap-ndp']
    # sage_methods = ['sage-inf', 'sage-edp', 'sage-ndp']
    # mlp_methods  = ['mlp', 'mlp-dp']

    inf_methods  = ['progap-inf'
                    # , 'gap-inf'
                    ]
    edp_methods  = ['progap-edp', 
                    # 'gap-edp', 'mlp'
                    ]
    ndp_methods  = ['progap-ndp', 
                    # 'gap-ndp', 'mlp-dp'
                    ]

    all_methods  = inf_methods + edp_methods + ndp_methods
    hparams = {dataset: {method: {} for method in all_methods} for dataset in datasets}

    for dataset in datasets:
        # For ProGAP methods
        for method in progap_methods:
            hparams[dataset][method]['encoder_layers'] = [1, 2]
            hparams[dataset][method]['head_layers'] = 1
            hparams[dataset][method]['jk'] = 'cat'
            hparams[dataset][method]['stages'] = [2, 3, 4, 5, 6]
            hparams[dataset][method]['layerwise'] = False
        
        # # For GAP methods
        # for method in gap_methods:
        #     hparams[dataset][method]['encoder_layers'] = 2
        #     hparams[dataset][method]['base_layers'] = 1
        #     hparams[dataset][method]['head_layers'] = 1
        #     hparams[dataset][method]['combine'] = 'cat'
        #     hparams[dataset][method]['hops'] = [1, 2, 3, 4, 5]
        
        # # For SAGE methods
        # for method in sage_methods:
        #     hparams[dataset][method]['base_layers'] = 2
        #     hparams[dataset][method]['head_layers'] = 1
        #     if method != 'sage-ndp':
        #         hparams[dataset][method]['mp_layers'] = [1, 2, 3, 4, 5]
        
        # # For MLP methods
        # for method in mlp_methods:
        #     hparams[dataset][method]['num_layers'] = 3
        
        # For graph-based NDP methods
        for method in set(ndp_methods) - {'mlp-dp'}:
            hparams[dataset][method]['max_degree'] = max_degree[dataset]
        
        # For all methods
        for method in all_methods:
            hparams[dataset][method]['hidden_dim'] = 16
            hparams[dataset][method]['activation'] = 'selu'
            hparams[dataset][method]['optimizer'] = 'adam'
            hparams[dataset][method]['learning_rate'] = [0.01, 0.05]
            hparams[dataset][method]['repeats'] = 10
            if method in ndp_methods:
                hparams[dataset][method]['max_grad_norm'] = 1.0
                hparams[dataset][method]['epochs'] = [5, 10]
                hparams[dataset][method]['batch_size'] = batch_size[dataset]
            else:
                hparams[dataset][method]['batch_norm'] = True
                hparams[dataset][method]['epochs'] = 100
                hparams[dataset][method]['batch_size'] = 'full'

    progress = Progress(
        *Progress.get_default_columns(),
        "[cyan]{task.fields[registered]}[/cyan] jobs registered",
        console=console,
    )
    task = progress.add_task('generating jobs', total=None, registered=0)

    with progress:
        # ### Accuracy/Privacy Trade-off
        for dataset in datasets:
            for method in all_methods:
                params = {}
                if method in ndp_methods:
                    params['epsilon'] = [2, 4, 8, 16, 32]
                elif method in set(edp_methods) - {'mlp'}:
                    params['epsilon'] = [0.25, 0.5, 1, 2, 4]
                    
                registry.register(
                    'train.py',
                    method, 
                    dataset=dataset,
                    **params, 
                    **hparams[dataset][method]
                )

                progress.update(task, registered=len(registry.job_list))

        # ### Convergence
        for dataset in datasets:
            for method in progap_methods:
                params = {**hparams[dataset][method]}
                params['repeats'] = 1
                params['stages'] = 6

                if method in ndp_methods:
                    params['epsilon'] = 8
                    params['epochs'] = 10
                elif method in set(edp_methods) - {'mlp'}:
                    params['epsilon'] = 1
                    params['epochs'] = 100
                
                params['log_all'] = True
                registry.register(
                    'train.py',
                    method, 
                    dataset=dataset,
                    **params,
                )

                progress.update(task, registered=len(registry.job_list))

        # ### Progressive vs. Layer-wise
        for dataset in datasets:
            for method in progap_methods:
                params = {**hparams[dataset][method]}
                
                if method in ndp_methods:
                    params['epsilon'] = 8
                elif method in set(edp_methods) - {'mlp'}:
                    params['epsilon'] = 1
                
                params['layerwise'] = True
                registry.register(
                    'train.py',
                    method, 
                    dataset=dataset,
                    **params,
                )

                params['layerwise'] = False
                registry.register(
                    'train.py',
                    method, 
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

    registry = WandBJobRegistry(
        entity=wandb_config['username'], 
        project=wandb_config['project']['train']
    )

    # with console.status('pulling jobs from WandB'):
    registry.pull()

    # with console.status('generating job commands'):
    jobs = create_train_commands(registry)
    
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
