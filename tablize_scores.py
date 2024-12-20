import os
import csv
import pandas as pd

# Directory configurations
RESULTS_DIR = "results"
SCORES_DIR = "scores"
SCORE_TYPES = ["ROC_AUC_SCORE", "F1_SCORE"]
ACTIVATION_FUNCTIONS = ["softmax", "sigmoid"]

# Dataset names for parsing
data_names = [
    'sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4',
    'bpic2011_f1', 'bpic2011_f2', 'bpic2011_f3', 'bpic2011_f4',
    'bpic2012_accepted', 'bpic2012_declined', 'bpic2012_cancelled',
    'production'
]

def parse_results():
    summary_data = {score_type: [] for score_type in SCORE_TYPES}

    for file_name in os.listdir(RESULTS_DIR):
        if not file_name.endswith(".csv"):
            continue

        # Parse file components
        dataset_name = next((name for name in data_names if name in file_name), None)
        if not dataset_name:
            continue

        remaining_name = file_name.split(f"{dataset_name}")[-1]
        activation_function = next((af for af in ACTIVATION_FUNCTIONS if af in remaining_name), None)

        if activation_function:
            parts = remaining_name.split(activation_function)
            encoding_type = parts[0].strip('_')
            mfs_type = parts[1].replace(".csv", "").strip('_')
        else:
            continue

        file_path = os.path.join(RESULTS_DIR, file_name)
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 2:
                    continue
                score_type = row[0].split(' ')[0]  # Extract score type (e.g., ROC_AUC_SCORE)
                if score_type in SCORE_TYPES:
                    try:
                        score_value = float(row[1])
                    except ValueError:
                        continue

                    # Append to summary data
                    summary_data[score_type].append({
                        'dataset_name': dataset_name,
                        'score_value': score_value,
                        'encoding_type': encoding_type,
                        'activation_function': activation_function,
                        'mfs_type': mfs_type
                    })

    # Generate CSV tables for each dataset and score type
    for score_type in SCORE_TYPES:
        summary_df = pd.DataFrame(summary_data[score_type])
        if summary_df.empty:
            continue

        for dataset_name in summary_df['dataset_name'].unique():
            dataset_df = summary_df[summary_df['dataset_name'] == dataset_name]
            dataset_dir = os.path.join(SCORES_DIR, score_type, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            for fixed_type, variable_1, variable_2 in [
                ('encoding_type', 'activation_function', 'mfs_type'),
                ('activation_function', 'encoding_type', 'mfs_type'),
                ('mfs_type', 'encoding_type', 'activation_function')
            ]:
                for fixed_value in dataset_df[fixed_type].unique():
                    filtered_df = dataset_df[dataset_df[fixed_type] == fixed_value]

                    pivot_df = filtered_df.pivot(
                        index=variable_1,
                        columns=variable_2,
                        values='score_value'
                    )

                    output_file = os.path.join(dataset_dir, f"{fixed_type}_{fixed_value}.csv")
                    pivot_df.to_csv(output_file)

        # Generate top-level summary CSV for the best scores
        best_scores = dataset_df.loc[dataset_df.groupby('dataset_name')['score_value'].idxmax()]
        best_scores = best_scores[['dataset_name', 'score_value', 'encoding_type', 'activation_function', 'mfs_type']]
        best_scores.to_csv(os.path.join(SCORES_DIR, score_type, "best_scores.csv"), index=False)

if __name__ == "__main__":
    parse_results()
