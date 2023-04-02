import os
import wandb
import pandas as pd
import numpy as np
from itertools import product
from core import console
from rich.progress import track


class WandBJobRegistry:
    """Job registry utility based on WandB

    Args:
        entity (str): Name of the entity (e.g., the username).
        project (str): Project name.
    """
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self.job_list = []
        self.df_stored_jobs = pd.DataFrame()

    def pull(self):
        """Pull all runs from WandB"""
        api = wandb.Api()
        projects = [project.name for project in api.projects(entity=self.entity)]

        if self.project in projects:
            runs = api.runs(f"{self.entity}/{self.project}", per_page=2000)
            config_list = []
            for run in track(runs, description='pulling jobs from wandb server', console=console):
                config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})

            self.df_stored_jobs = pd.DataFrame.from_dict(config_list)
            if 'epsilon' in self.df_stored_jobs.columns:
                self.df_stored_jobs['epsilon'] = self.df_stored_jobs['epsilon'].astype(float)
            
            self.df_stored_jobs.drop_duplicates(inplace=True)
    
    def register(self, main_file, method, level, **params) -> list[str]:
        """Register jobs to the registry.
        This method will generate all possible combinations of the parameters and
        create a list of jobs to run. The job commands are stored in the registry
        if the corresponding runs are not already present in the WandB server.

        Args:
            main_file (str): Path to the main executable python file.
            method (str): Name of the method to run.
            level (str): Privacy level of the method.
            **params (dict): Dictionary of parameters to sweep over.

        Returns:
            list[str]: List of jobs to run.
        """

        # convert all values to tuples
        for key, value in params.items():
            if not (isinstance(value, list) or isinstance(value, tuple)):
                params[key] = (value,)
        
        # rule out already existing jobs
        param_keys = list(params.keys())
        param_values = list(product(*params.values()))
        df_new_configs = pd.DataFrame(param_values, columns=param_keys)
        df_new_configs['method'] = method
        df_new_configs['level'] = level
        
        if self.df_stored_jobs.empty or (set(param_keys) - set(self.df_stored_jobs.columns)):
            df_out_configs = df_new_configs
        else:
            df_out_configs = df_new_configs.merge(self.df_stored_jobs, how='left', indicator=True)
            df_out_configs = df_out_configs[df_out_configs['_merge'] == 'left_only']
            df_out_configs = df_out_configs[df_new_configs.columns]

        # generate job commands
        jobs = []
        configs = df_out_configs.to_dict('records')
        for config in configs:
            config.pop('method', None)
            config.pop('level', None)
            args = f" {method} {level} "
            options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
            command = f'python {main_file} {args} {options} --logger wandb --project {self.project}'
            command = ' '.join(command.split())
            jobs.append(command)

        self.job_list += jobs
        return jobs

    def save(self, path: str, sort=False, shuffle=False):
        """Save the job list to a file.

        Args:
            path (str): Path to the file.
            sort (bool, optional): Sort the job list. Defaults to False.
            shuffle (bool, optional): Shuffle the job list. Defaults to False.
        """

        assert not (sort and shuffle), 'cannot sort and shuffle at the same time'

        # remove duplicates
        self.job_list = list(dict.fromkeys(self.job_list))

        if sort:
            jobs = sorted(self.job_list)
        elif shuffle:
            jobs = np.random.choice(self.job_list, len(self.job_list), replace=False)
        else:
            jobs = self.job_list

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for item in jobs:
                print(item, file=file)
