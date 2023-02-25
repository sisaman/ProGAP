import os
from pathlib import Path
import subprocess
import dask
from dask.distributed import Client
from dask_jobqueue import JobQueueCluster
from dask.distributed import as_completed
from rich.progress import Progress
from subprocess import CalledProcessError
from class_resolver import ClassResolver
from core import console, Console

dask.config.set({"jobqueue.sge.walltime": None})
dask.config.set({"distributed.worker.memory.target": False})    # Avoid spilling to disk
dask.config.set({"distributed.worker.memory.spill": False})     # Avoid spilling to disk
dask.config.set({'distributed.scheduler.allowed-failures': 99}) # Allow workers to fail


class JobScheduler:
    """Distributed job scheduler using Dask.

    Args:
        job_file (str): The job file to schedule. Each line is a job command.
        scheduler (str, optional): The scheduler to use. Options are 'htcondor', 
            'lsf', 'moab', 'oar', 'pbs', 'sge', 'slurm'. Defaults to 'sge'.
    """
    def __init__(self, job_file: str, scheduler: str = 'sge'):
        assert scheduler in self.cluster_resolver.options, f'Invalid scheduler: {scheduler}'
        self.scheduler = scheduler
        self.file = job_file
        path = os.path.realpath(self.file)
        self.name = Path(path).stem
        self.job_dir = os.path.join(os.path.dirname(path), self.name)

        with open(self.file) as jobs_file:
            self.job_list = jobs_file.read().splitlines()

    def submit(self):
        """Submit the job file to the cluster.

        Returns:
            list[str]: The list of failed job commands.
        """

        total = len(self.job_list)
        progress = SchedulerProgress(total=total, console=console)

        num_failed_jobs = 0
        failures_dir = os.path.join(self.job_dir, 'failures')
        os.makedirs(failures_dir, exist_ok=True)

        cluster: JobQueueCluster = self.cluster_resolver.make(
            self.scheduler,
            job_name=f'dask-{self.name}',
            log_directory=os.path.join(self.job_dir, 'logs'),
        )
        
        with cluster:   
            cluster.adapt(minimum=min(160, total))  # 160 is the max number of gpus on the cluster
            
            with Client(cluster) as client:
                with progress:
                    futures = client.map(
                        lambda cmd: subprocess.run(args=cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT),
                        self.job_list,
                    )
                    for future in as_completed(futures, with_results=False):
                        try:
                            future.result()
                            progress.update(failed=False)
                        except CalledProcessError as e:
                            num_failed_jobs += 1
                            job_cmd = ' '.join(e.cmd)
                            failed_job_output = e.output.decode()
                            with open(os.path.join(failures_dir, f'{num_failed_jobs}.log'), 'w') as f:
                                print(job_cmd, end='\n\n', file=f)
                                print(failed_job_output, file=f)
                            progress.update(failed=True)

    cluster_resolver = ClassResolver.from_subclasses(JobQueueCluster, suffix='Cluster')



class SchedulerProgress:
    """Progress bar for the scheduler.

    Args:
        total (int): The total number of jobs.
        console (Console): The rich console to use.
    """
    def __init__(self, total: int, console: Console):
        self.finished = 0
        self.failed = 0
        self.remaining = total

        self.bar = Progress(
            *Progress.get_default_columns(),
            "Completed: [green]{task.fields[finished]}",
            "Failed: [red]{task.fields[failed]}",
            "Remaining: [blue]{task.fields[remaining]}",
            console=console,
        )

        self.task = self.bar.add_task(
            description="Running jobs",
            finished=self.finished,
            failed=self.failed,
            remaining=self.remaining,
            total=total
        )

    def update(self, failed: bool):
        """Update the progress bar.

        Args:
            failed (bool): Whether the job failed or not.
        """
        self.finished += int(not failed)
        self.failed += int(failed)
        self.remaining -= 1

        self.bar.update(
            self.task,
            advance=1,
            finished=self.finished,
            failed=self.failed,
            remaining=self.remaining,
            refresh=True
        )

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, *args):
        return self.bar.__exit__(*args)
