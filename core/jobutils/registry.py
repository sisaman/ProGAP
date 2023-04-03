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
        self.df_job_cmds = pd.DataFrame()
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
    
    def register(self, main_file, method, level, **params):
        """Register jobs to the registry.
        This method will generate all possible combinations of the parameters and
        create a list of jobs to run. The job commands are stored in the registry
        if the corresponding runs are not already present in the WandB server.

        Args:
            main_file (str): Path to the main executable python file.
            method (str): Name of the method to run.
            level (str): Privacy level of the method.
            **params (dict): Dictionary of parameters to sweep over.
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
        def make_cmd(row):
            method = row['method']
            level = row['level']
            row = row.drop(['method', 'level'])
            args = f" {method} {level} "
            options = ' '.join([f' --{param} {value} ' for param, value in row.items()])
            command = f'python {main_file} {args} {options} --logger wandb --project {self.project}'
            command = ' '.join(command.split())
            return command

        df_out_configs['cmd'] = df_out_configs.apply(make_cmd, axis=1)
        self.df_job_cmds = pd.concat([self.df_job_cmds, df_out_configs], ignore_index=True)

    def save(self, path: str, sort=False, shuffle=False) -> int:
        """Save the job list to a file.

        Args:
            path (str): Path to the file.
            sort (bool, optional): Sort the job list. Defaults to False.
            shuffle (bool, optional): Shuffle the job list. Defaults to False.

        Returns:
            int: Number of jobs saved.
        """

        assert not (sort and shuffle), 'cannot sort and shuffle at the same time'

        # remove duplicates
        columns = self.df_job_cmds.columns.drop('cmd')
        self.df_job_cmds.drop_duplicates(subset=columns, inplace=True)
        job_cmds = self.df_job_cmds['cmd'].tolist()

        if sort:
            job_cmds = sorted(job_cmds)
        elif shuffle:
            job_cmds = np.random.choice(job_cmds, len(job_cmds), replace=False)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for item in job_cmds:
                print(item, file=file)

        return len(job_cmds)
