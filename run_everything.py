import os
import subprocess
import multiprocessing
import logging
import time
import gpustat
from queue import Queue
from constants import BASE_MODEL_CONFIG, dataset_names_to_folders, clean_string

def get_gpu_memory_usage(gpu_id):
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_memory_usage = gpu_stats.jsonify()["gpus"][gpu_id]["memory.used"]
        return gpu_memory_usage
    except:
        logging.exception(f"Error retrieving GPU memory usage for GPU {gpu_id}")
        return None

def run_experiment(gpu_id, backbone, dataset):
    log_file = f"run_logs/{clean_string(backbone)}_{clean_string(dataset)}.log"
    with open(log_file, "w") as f:
        command = f"python trainer_wandb.py --base_model {backbone} --dataset_name {dataset} --gpu {gpu_id}"
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=f, universal_newlines=True)
        process.wait()
    return process.returncode

def worker(gpu_id, queue, completed_jobs, failed_jobs, max_concurrent_jobs, max_memory_usage):
    running_jobs = []
    while True:
        if len(running_jobs) >= max_concurrent_jobs:
            time.sleep(1)
            running_jobs = [job for job in running_jobs if job.is_alive()]
            continue

        experiment = queue.get()
        if experiment is None:
            break

        backbone, dataset = experiment
        logging.info(f"Running experiment: backbone={backbone}, dataset={dataset} on GPU {gpu_id}")

        gpu_memory_usage = get_gpu_memory_usage(gpu_id)
        if gpu_memory_usage is None:
            logging.warning(f"Failed to retrieve GPU memory usage for GPU {gpu_id}. Falling back to max_concurrent_jobs limit.")
        elif gpu_memory_usage >= max_memory_usage:
            logging.warning(f"GPU {gpu_id} memory usage ({gpu_memory_usage} MB) exceeds the limit ({max_memory_usage} MB). Waiting...")
            queue.put(experiment)
            queue.task_done()
            time.sleep(5)
            continue

        job = multiprocessing.Process(target=run_job, args=(gpu_id, backbone, dataset, completed_jobs, failed_jobs))
        job.start()
        running_jobs.append(job)
        queue.task_done()

def run_job(gpu_id, backbone, dataset, completed_jobs, failed_jobs):
    try:
        returncode = run_experiment(gpu_id, backbone, dataset)
        if returncode == 0:
            #log that experiment completed
            logging.info(f"Experiment completed: backbone={backbone}, dataset={dataset}")
            completed_jobs.append((backbone, dataset))
        else:
            #log that experiment failed
            logging.info(f"Experiment failed: backbone={backbone}, dataset={dataset}")
            failed_jobs.append((backbone, dataset, returncode))
    except Exception as e:
        logging.exception(f"Error running experiment: backbone={backbone}, dataset={dataset}")
        failed_jobs.append((backbone, dataset, -1))

def main():
    num_gpus = 4
    max_concurrent_jobs = 16
    max_memory_usage = 70 * 1024  # 40 GB
    experiments = [(backbone, dataset) for backbone in BASE_MODEL_CONFIG.keys() for dataset in dataset_names_to_folders.keys()]
    num_experiments = len(experiments)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    completed_jobs = manager.list()
    failed_jobs = manager.list()

    for experiment in experiments:
        queue.put(experiment)

    processes = []
    for gpu_id in range(num_gpus):
        process = multiprocessing.Process(target=worker, args=(gpu_id, queue, completed_jobs, failed_jobs, max_concurrent_jobs, max_memory_usage))
        process.start()
        processes.append(process)

    logging.info(f"Total experiments: {num_experiments}")
    while True:
        completed_count = len(completed_jobs)
        failed_count = len(failed_jobs)
        logging.info(f"Completed: {completed_count}, Failed: {failed_count}, Remaining: {num_experiments - completed_count - failed_count}")
        if completed_count + failed_count == num_experiments:
            break
        time.sleep(10)

    for _ in range(num_gpus):
        queue.put(None)

    for process in processes:
        process.join()

    logging.info(f"Completed {len(completed_jobs)} out of {num_experiments} experiments.")
    if failed_jobs:
        logging.info(f"Failed experiments: {len(failed_jobs)}")
        for experiment, returncode in failed_jobs:
            logging.error(f"Experiment: backbone={experiment[0]}, dataset={experiment[1]}")
            logging.error(f"Return Code: {returncode}")

    retry_failed_experiments(failed_jobs, completed_jobs, num_gpus, max_concurrent_jobs, max_memory_usage)

def retry_failed_experiments(failed_jobs, completed_jobs, num_gpus, max_concurrent_jobs, max_memory_usage):
    logging.info("Retrying failed experiments...")
    queue = multiprocessing.Manager().Queue()
    for experiment, _, _ in failed_jobs:
        if experiment not in completed_jobs:
            queue.put(experiment)

    processes = []
    for gpu_id in range(num_gpus):
        process = multiprocessing.Process(target=worker, args=(gpu_id, queue, completed_jobs, failed_jobs, max_concurrent_jobs, max_memory_usage))
        process.start()
        processes.append(process)

    for _ in range(num_gpus):
        queue.put(None)

    for process in processes:
        process.join()

    logging.info(f"Completed {len(completed_jobs)} out of {len(completed_jobs) + len(failed_jobs)} experiments after retry.")
    if failed_jobs:
        logging.info(f"Failed experiments after retry: {len(failed_jobs)}")
        for experiment, returncode in failed_jobs:
            logging.error(f"Experiment: backbone={experiment[0]}, dataset={experiment[1]}")
            logging.error(f"Return Code: {returncode}")

if __name__ == "__main__":
    os.makedirs("run_logs", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()