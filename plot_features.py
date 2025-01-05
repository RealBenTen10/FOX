import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_columns_sel(dataset_name):
    if dataset_name == 'sepsis_cases_1':
        return ['Diagnose', 'mean_open_cases', 'Age', 'std_Leucocytes', 'std_CRP']
    elif dataset_name == 'sepsis_cases_2':
        return ['Diagnose', 'mean_open_cases', 'mean_hour', 'DisfuncOrg']
    elif dataset_name == 'sepsis_cases_4':
        return ['Diagnose', 'mean_open_cases', 'Age', 'org:group_E', 'std_CRP', 'DiagnosticECG']
    elif dataset_name == 'bpic2011_f1':
        return ['Diagnosis Treatment Combination ID', 'mean_open_cases', 'Diagnosis', 'Activity code_376400.0']
    elif dataset_name == 'bpic2011_f2':
        return ['Diagnosis Treatment Combination ID', 'Diagnosis', 'Diagnosis code', 'mean_open_cases',
                'Activity code_376400.0', 'Age', 'Producer code_CHE1']
    elif dataset_name == 'bpic2011_f3':
        return ['Diagnosis Treatment Combination ID', 'Diagnosis', 'mean_open_cases', 'Diagnosis code',
                'std_event_nr', 'mean_event_nr']
    elif dataset_name == 'bpic2011_f4':
        return ['Diagnosis Treatment Combination ID', 'Treatment code']
    elif dataset_name == 'bpic2012_accepted':
        return ['AMOUNT_REQ', 'Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE',
                'Activity_W_Valideren aanvraag-START']
    elif dataset_name == 'bpic2012_declined':
        return ['AMOUNT_REQ', 'Activity_A_PARTLYSUBMITTED-COMPLETE', 'Activity_A_PREACCEPTED-COMPLETE',
                'Activity_A_DECLINED-COMPLETE', 'Activity_W_Completeren aanvraag-SCHEDULE', 'mean_open_cases']
    elif dataset_name == 'bpic2012_cancelled':
        return ['Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE',
                'Activity_W_Valideren aanvraag-START', 'AMOUNT_REQ', 'Activity_W_Valideren aanvraag-COMPLETE',
                'Activity_A_CANCELLED-COMPLETE']
    elif dataset_name == 'production':
        return ['Work_Order_Qty', 'Activity_Turning & Milling - Machine 4', 'Resource_ID0998', 'Resource_ID4794',
                'Resource.1_Machine 4 - Turning & Milling']
    else:
        return []

def plot_feature_distribution(dataset_name, dataset_path):
    # Get the selected columns
    columns_sel = get_columns_sel(dataset_name)
    if not columns_sel:
        print(f"No columns selected for dataset: {dataset_name}")
        return

    # Load train and test datasets
    train_file = os.path.join(dataset_path, f"{dataset_name}_train.csv")
    test_file = os.path.join(dataset_path, f"{dataset_name}_test.csv")

    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    # Combine train and test datasets
    data = pd.concat([train_data, test_data])

    # Filter columns
    data = data[columns_sel]

    # Plot each feature
    output_dir = os.path.join('plots')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for column in columns_sel:
        if column not in data.columns:
            print(f"Column {column} not found in dataset {dataset_name}")
            continue

        # Drop missing values
        feature_data = data[column].dropna()

        # Skip non-numeric columns
        if not np.issubdtype(feature_data.dtype, np.number):
            print(f"Skipping non-numeric column: {column}")
            continue

        # Compute value occurrences before normalization
        value_counts = feature_data.value_counts(normalize=True).sort_index()
        values = value_counts.index.values
        relative_occurrences = value_counts.values

        # Normalize the values between 0 and 1
        min_val = values.min()
        max_val = values.max()
        normalized_values = (values - min_val) / (max_val - min_val)

        # Plot the relative occurrences of normalized values
        plt.plot(normalized_values, relative_occurrences, label=column)

    plt.title(f"Feature Distributions ({dataset_name})")
    plt.xlabel("Normalized Value (0-1)")
    plt.ylabel("Relative Occurrence")
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.5)

    # Save the plot for the dataset
    plot_file = os.path.join(output_dir, f"{dataset_name}_distribution.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Saved plot for dataset {dataset_name} to {plot_file}")

if __name__ == "__main__":
    dataset_root = "dataset"
    dataset_names = [
        'sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4',
        'bpic2011_f1', 'bpic2011_f2', 'bpic2011_f3', 'bpic2011_f4',
        'bpic2012_accepted', 'bpic2012_declined', 'bpic2012_cancelled', 'production'
    ]

    for dataset_name in dataset_names:
        dataset_path = os.path.join(dataset_root, dataset_name)
        plot_feature_distribution(dataset_name, dataset_path)
