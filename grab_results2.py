import os
import re
import pandas as pd
from collections import defaultdict

def process_log_file(log_file):
    """
    Process a single log file and extract the relevant metrics.
    
    Args:
        log_file (str): Path to the log file.
    
    Returns:
        tuple: A tuple containing the model name, training dataset, and a list of tuples
               containing the extracted metrics in the format (testing dataset, metric_name, value).
    """
    with open(log_file, 'r') as file:
        content = file.read()
    
    file_name = os.path.basename(log_file)
    model_name = '_'.join(file_name.split('_')[:-1])
    training_dataset = file_name.split('_')[-1].split('.')[0]

    #remove "dataset" from the training dataset name
    training_dataset = training_dataset.replace("dataset", "")
    
    run_summary_pattern = r'wandb: Run summary:(.*?)(?:\n\s*\n|\Z)'
    run_summary_match = re.search(run_summary_pattern, content, re.DOTALL)
    
    if run_summary_match:
        run_summary = run_summary_match.group(1)
        metric_pattern = r'wandb:\s+(\w+)_(\w+)Dataset_(\w+)\s+(-?\d+\.\d+|\d+)'
        metrics = re.findall(metric_pattern, run_summary)
        
        processed_metrics = []
        for metric in metrics:
            _, testing_dataset, metric_name, value = metric
            processed_metrics.append((testing_dataset.lower(), metric_name, float(value)))
        
        return model_name, training_dataset, processed_metrics
    else:
        print(f"Error: 'wandb: Run summary:' section not found in {log_file}")
        return None, None, []

def organize_metrics(accumulated_metrics, datasets):
    """
    Organize the accumulated metrics into a dictionary of DataFrames.
    
    Args:
        accumulated_metrics (dict): A dictionary containing the accumulated metrics for each model and metric combination.
        datasets (list): A sorted list of unique dataset names.
    
    Returns:
        dict: A dictionary where the keys are metric names and the values are DataFrames containing the metric values.
    """
    metric_tables = {}
    
    for metric_name, model_metrics in accumulated_metrics.items():
        if metric_name not in metric_tables:
            metric_tables[metric_name] = pd.DataFrame(index=datasets, columns=datasets, dtype=float)
            metric_tables[metric_name].fillna(float('NaN'), inplace=True)
        
        for training_dataset, testing_dataset, value in model_metrics:
            metric_tables[metric_name].loc[training_dataset, testing_dataset] = value
    
    return metric_tables

def save_tables(metric_tables, output_dir):
    """
    Save the metric tables as CSV files in the specified output directory.
    
    Args:
        metric_tables (dict): A dictionary containing the metric tables.
        output_dir (str): The output directory where the CSV files will be saved.
    """
    for model_name, model_tables in metric_tables.items():
        model_dir = os.path.join(output_dir, model_name)
        
        try:
            os.makedirs(model_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {model_dir}: {str(e)}")
            continue
        
        for metric_name, table in model_tables.items():
            file_path = os.path.join(model_dir, f"{metric_name}.csv")
            try:
                table.to_csv(file_path, index_label='Training Dataset')
                print(f"Saved {metric_name} table for {model_name} at {file_path}")
            except IOError as e:
                print(f"Error saving {metric_name} table for {model_name}: {str(e)}")

def process_log_files(log_directory, output_dir):
    """
    Process all the log files in the specified directory and generate the metric tables.
    
    Args:
        log_directory (str): The directory containing the log files.
        output_dir (str): The output directory where the CSV files will be saved.
    """
    if not os.path.exists(log_directory):
        print(f"Error: Log directory {log_directory} does not exist.")
        return
    
    log_files = [file for file in os.listdir(log_directory) if file.endswith('.log')]
    
    if not log_files:
        print(f"No log files found in {log_directory}")
        return
    
    datasets = set()
    accumulated_metrics = defaultdict(lambda: defaultdict(list))
    
    for log_file in log_files:
        log_file_path = os.path.join(log_directory, log_file)
        
        try:
            model_name, training_dataset, metrics = process_log_file(log_file_path)
            if model_name is not None:
                datasets.add(training_dataset)
                for testing_dataset, metric_name, value in metrics:
                    datasets.add(testing_dataset)
                    accumulated_metrics[metric_name][model_name].append((training_dataset, testing_dataset, value))
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
    
    datasets = sorted(datasets)
    
    metric_tables = {}
    for metric_name, model_metrics in accumulated_metrics.items():
        metric_tables[metric_name] = organize_metrics(model_metrics, datasets)
    
    save_tables(metric_tables, output_dir)
    
    print("\nSummary:")
    print(f"Number of log files processed: {len(log_files)}")
    print(f"Number of unique datasets: {len(datasets)}")
    print("Metrics processed:")
    for metric_name in metric_tables.keys():
        print(f"- {metric_name}")

# Example usage
log_directory = 'run_logs'
output_dir = 'collected_results2'

process_log_files(log_directory, output_dir)