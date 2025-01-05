import os
import csv
import pandas as pd

# Function to parse the CSV file and extract scores
def parse_results_file(file_path):
    roc_auc, f1_score = None, None
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if row and row[0] == 'ROC_AUC_SCORE weighted':
                roc_auc = round(float(row[1]), 3)
            elif row and row[0] == 'F1_SCORE weighted':
                f1_score = round(float(row[1]), 3)
    return roc_auc, f1_score

# Function to process all files in a folder
def process_results_folder(folder_path):
    roc_auc_table = {}
    f1_score_table = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('_results.csv'):
            # Extract dataset and member function from file name
            base_name = file_name.replace("_one_hot_softmax", "").replace("_Sugeno", "")[:-12]  # Remove '_results.csv'
            dataset_name, member_function = base_name.rsplit('_', 1)

            # Parse the file to get scores
            file_path = os.path.join(folder_path, file_name)
            roc_auc, f1_score = parse_results_file(file_path)

            # Populate the tables
            if dataset_name not in roc_auc_table:
                roc_auc_table[dataset_name] = {}
                f1_score_table[dataset_name] = {}

            roc_auc_table[dataset_name][member_function] = roc_auc
            f1_score_table[dataset_name][member_function] = f1_score

    return roc_auc_table, f1_score_table

# Function to determine the best function for each dataset
def add_best_column(score_table):
    best_column = {}
    for dataset, scores in score_table.items():
        if scores:
            # Filter out None values
            valid_scores = {key: value for key, value in scores.items() if value is not None}
            if valid_scores:  # Ensure there are valid scores
                best_function = max(valid_scores, key=valid_scores.get)
                best_column[dataset] = best_function
            else:
                best_column[dataset] = None  # No valid scores
    return best_column

# Function to format DataFrame for LaTeX output with bold max values and escape special characters
def format_for_latex(df):
    def escape_latex(text):
        if isinstance(text, str):
            return (text.replace("_", "\\_")
                        .replace("%", "\\%")
                        .replace("&", "\\&")
                        .replace("#", "\\#")
                        .replace("$", "\\$")
                        .replace("{", "\\{")
                        .replace("}", "\\}"))
        return text

    def bold_max(row):
        numeric_values = [(i, val) for i, val in enumerate(row) if isinstance(val, (int, float))]
        if numeric_values:
            max_value = max(val for i, val in numeric_values)
            for i, val in numeric_values:
                if val == max_value:
                    row[i] = f"\\textbf{{{row[i]}}}"
        return row

    # Escape LaTeX characters in headers and indices
    df.columns = [escape_latex(col) for col in df.columns]
    df.index = [escape_latex(idx) for idx in df.index]

    # Escape LaTeX characters in the data
    df = df.applymap(escape_latex)

    # Bold the maximum numeric values in each row
    df = df.apply(lambda row: bold_max(row), axis=1)

    return df

# Function to convert tables to dataframes and save as CSV
def save_tables_as_csv(roc_auc_table, f1_score_table, output_folder):
    # Convert to DataFrame
    roc_auc_df = pd.DataFrame(roc_auc_table).transpose()
    f1_score_df = pd.DataFrame(f1_score_table).transpose()

    # Save to raw CSV
    roc_auc_df.to_csv(os.path.join(output_folder, 'roc_auc_scores.csv'), index=True)
    f1_score_df.to_csv(os.path.join(output_folder, 'f1_scores.csv'), index=True)

    # Format DataFrames for LaTeX
    roc_auc_df = format_for_latex(roc_auc_df)
    f1_score_df = format_for_latex(f1_score_df)

    # Save to LaTeX-compatible CSV
    roc_auc_df.to_csv(os.path.join(output_folder, 'roc_auc_scores_latex.csv'), index=True)
    f1_score_df.to_csv(os.path.join(output_folder, 'f1_scores_latex.csv'), index=True)

# Main execution
def main():
    input_folder = './results/'  # Replace with your folder path
    output_folder = './scores/'  # Replace with your folder path

    roc_auc_table, f1_score_table = process_results_folder(input_folder)
    save_tables_as_csv(roc_auc_table, f1_score_table, output_folder)

if __name__ == "__main__":
    main()
