import numpy as np
import load_weights
import torch
from PIL import Image
import math
import plotly.express as px
import plotly.subplots as sp
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='FOX')

# Lade das Bild aus dem gleichen Ordner wie das Skript
image = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fox.png'))
st.image(image)

# Dataset-Auswahl
dataset_name = st.sidebar.selectbox('Choose Dataset',
    ('sepsis_cases_1', 'sepsis_cases_2', 'sepsis_cases_4',
     'bpic2011_f1', 'bpic2011_f2', 'bpic2011_f3', 'bpic2011_f4',
     'bpic2012_accepted', 'bpic2012_cancelled', 'bpic2012_declined', 'production')
    )

st.title('Performace on ' + str(dataset_name) + ' event log')

# Spaltenauswahl basierend auf Dataset
if dataset_name == 'sepsis_cases_1':
    columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'std_Leucocytes', 'std_CRP', 'Classification']
elif dataset_name == 'sepsis_cases_2':
    columns_sel = ['Diagnose', 'mean_open_cases', 'mean_hour', 'DisfuncOrg', 'Classification']
elif dataset_name == 'sepsis_cases_4':
    columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'org:group_E', 'std_CRP', 'DiagnosticECG', 'Classification']
elif dataset_name == 'bpic2011_f1':
    columns_sel = ['Diagnosis Treatment Combination ID', 'mean_open_cases', 'Diagnosis', 'Activity code_376400.0', 'Classification']
elif dataset_name == 'bpic2011_f2':
    columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'Diagnosis code', 'mean_open_cases', 'Activity code_376400.0', 'Age', 'Producer code_CHE1', 'Classification']
elif dataset_name == 'bpic2011_f3':
    columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'mean_open_cases', 'Diagnosis code', 'std_event_nr', 'mean_event_nr', 'Classification']
elif dataset_name == 'bpic2011_f4':
    columns_sel = ['Diagnosis Treatment Combination ID', 'Treatment code', 'Classification']
elif dataset_name == 'bpic2012_accepted':
    columns_sel = ['AMOUNT_REQ', 'Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START', 'Classification']
elif dataset_name == 'bpic2012_declined':
    columns_sel = ['AMOUNT_REQ', 'Activity_A_PARTLYSUBMITTED-COMPLETE', 'Activity_A_PREACCEPTED-COMPLETE', 'Activity_A_DECLINED-COMPLETE', 'Activity_W_Completeren aanvraag-SCHEDULE', 'mean_open_cases', 'Classification']
elif dataset_name == 'bpic2012_cancelled':
    columns_sel = ['Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START', 'AMOUNT_REQ', 'Activity_W_Valideren aanvraag-COMPLETE', 'Activity_A_CANCELLED-COMPLETE', 'Classification']
elif dataset_name == 'production':
    columns_sel = ['Work_Order_Qty', 'Activity_Turning & Milling - Machine 4', 'Resource_ID0998', 'Resource_ID4794', 'Resource.1_Machine 4 - Turning & Milling', 'Classification']

# Lade Test-Daten
df_test = pd.read_csv(f"dataset/{dataset_name}/{dataset_name}_test.csv", header=0, sep=',')

# Performance-Bereich
with st.form('performace'):
    submit_perf = st.form_submit_button('Get results')

if submit_perf:
    list_index_len, auc_list, auc_weight = load_weights.metrics(dataset_name, columns_sel)
    chart_data = pd.DataFrame(auc_list, columns=['AUC'])
    st.header('AUC: ' + str(round(auc_weight, 2)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list_index_len,
        y=auc_list,
        name=dataset_name,
        connectgaps=True
    ))
    st.plotly_chart(fig)

# Trace Outcome Prediction Bereich
st.title('Trace outcome prediction')
st.header('Please fill the boxes')

features_dictionary = dict.fromkeys(columns_sel[:len(columns_sel) - 1], "")

with st.form('Form1'):
    for k, v in features_dictionary.items():
        features_dictionary[k] = st.text_input(k, v)
        st.write(features_dictionary[k])
    submit = st.form_submit_button('Predict')

if submit:
    i = 0
    list_val = []
    while i < len(columns_sel[:len(columns_sel) - 1]):
        list_val.append(float(features_dictionary[columns_sel[i]]))
        i += 1

    list_val = np.array(list_val)
    model = torch.load(f'models/model_{dataset_name}.h5')
    pred = model(torch.Tensor([list_val]))
    pred2 = torch.argmax(pred, 1).detach().numpy()

    res = 'regular' if pred2 == 0 else 'deviant'
    st.text(f'This trace is {res}')

    rule, firerule, index_rule = load_weights.get_fire_strength(model, pred2)
    st.write('Because:')

    list_max_index = [f'RULE {r}' for r in index_rule]
    list_max_fire = [firerule.get(r) for r in index_rule]

    rule = rule.replace('mf0', 'low').replace('mf1', 'medium').replace('mf2', 'high')
    for i in range(len(columns_sel) - 1):
        rule = rule.replace(f'x{i}', columns_sel[i])

    list_exp = [c.replace('IF', '').split('THEN')[0] for c in rule.split('and')]
    st.write(list_exp)

    barplot = go.Figure(go.Bar(
        x=list_max_fire[:5],
        y=list_max_index[:5],
        orientation='h',
        ),
        layout=go.Layout(
            title="Top 5 rules",
            xaxis=dict(title="Fire strength"),
            yaxis=dict(title="Rules")
        )
    )
    st.plotly_chart(barplot)



with st.expander("ROC AUC Heatmap Analysis by Dataset"):
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agg_combined_resultsNew.csv')
    data = pd.read_csv(data_path)

    # Get unique values for axes
    unique_member_functions = sorted(data["MemberFunction"].unique())
    unique_loss_functions = sorted(data["LossFunction"].unique())

    # Create a table for best combinations
    best_combinations = []

    # Create a dynamic grid layout
    datasets = data["DatasetName"].unique()
    num_datasets = len(datasets)
    num_columns = 2
    num_rows = math.ceil(num_datasets / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(12, num_rows * 6), squeeze=False)
    axes = axes.flatten()

    for i, dataset in enumerate(datasets):
        filtered_data = data[data["DatasetName"] == dataset]

        # Find the best combination of MemberFunction and LossFunction
        best_row = filtered_data.loc[filtered_data["ROC_AUC_SCOREweighted"].idxmax()]
        best_combinations.append({
            "Dataset": dataset,
            "ROC AUC": best_row['ROC_AUC_SCOREweighted'],
            "Loss Function": best_row['LossFunction'],
            "Member Function": best_row['MemberFunction']
        })

        # Pivot data for heatmap, ensuring consistent axis ordering
        heatmap_data = filtered_data.pivot_table(
            index="MemberFunction",
            columns="LossFunction",
            values="ROC_AUC_SCOREweighted",
            aggfunc='mean',
        ).reindex(index=unique_member_functions, columns=unique_loss_functions)

        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[i])
        axes[i].set_title(f"ROC AUC Scores for {dataset}")
        axes[i].set_xlabel("Loss Function")
        axes[i].set_ylabel("Member Function")

    # Hide unused subplots if the grid has extra cells
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Display best combinations as a sortable table above heatmaps
    st.write("### Best Combinations Across Datasets")
    best_combinations_df = pd.DataFrame(best_combinations)
    sort_column = st.selectbox("Sort by column:", best_combinations_df.columns, index=1)
    ascending_order = st.radio("Order:", ["Ascending", "Descending"]) == "Ascending"
    st.write("### Combination Heatmap of ROC AUC per Dataset")
    best_combinations_df = best_combinations_df.sort_values(by=sort_column, ascending=ascending_order)

    st.dataframe(best_combinations_df)

    # Find the most common Member Function and Loss Function across all datasets
    most_common_member_function = best_combinations_df["Member Function"].mode()[0]
    most_common_loss_function = best_combinations_df["Loss Function"].mode()[0]

    # Display the most common Member Function and Loss Function
    st.write(f"**Most Common Member Function:** {most_common_member_function}")
    st.write(f"**Most Common Loss Function:** {most_common_loss_function}")

    # Display heatmaps
    st.pyplot(fig)

with st.expander("ROC AUC Analysis over traces length for Membership/ Loss Function Combinations"):
    combined_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_resultsNew.csv')
    if os.path.exists(combined_results_path):
        combined_results = pd.read_csv(combined_results_path)

        # Filteroptionen hinzufügen
        dataset_filter = st.selectbox("Filter by DatasetName:", combined_results["DatasetName"].unique())
        member_function_filter = st.multiselect("Filter by MemberFunction:", combined_results["MemberFunction"].unique(), default=combined_results["MemberFunction"].unique())

        # Daten filtern
        filtered_data = combined_results

        if dataset_filter:
            filtered_data = filtered_data[filtered_data["DatasetName"] == dataset_filter]

        if member_function_filter:
            filtered_data = filtered_data[filtered_data["MemberFunction"].isin(member_function_filter)]

        # Alle LossFunctions sicherstellen
        unique_loss_functions = combined_results["LossFunction"].unique()
        rows, cols = 2, 2  # Festes 2x2 Raster

        # Farbpalette für MemberFunctions erstellen
        unique_member_functions = combined_results["MemberFunction"].unique()
        color_map = {mf: color for mf, color in zip(unique_member_functions, px.colors.qualitative.Plotly)}

        # Subplots erstellen
        fig = sp.make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"LossFunction: {lf}" for lf in unique_loss_functions]
        )

        # Füge Scatterplots zu den Subplots hinzu
        legend_chart_idx = None
        max_legend_entries = 0

        for idx, loss_function in enumerate(unique_loss_functions):
            loss_data = filtered_data[filtered_data["LossFunction"] == loss_function]

            if not loss_data.empty:
                legend_entry_count = loss_data["MemberFunction"].nunique()
                if legend_entry_count > max_legend_entries:
                    max_legend_entries = legend_entry_count
                    legend_chart_idx = idx

        for idx, loss_function in enumerate(unique_loss_functions):
            loss_data = filtered_data[filtered_data["LossFunction"] == loss_function]

            if not loss_data.empty:
                for member_function in loss_data["MemberFunction"].unique():
                    member_data = loss_data[loss_data["MemberFunction"] == member_function]
                    fig.add_trace(
                        go.Scatter(
                            x=member_data["LENGHT"],
                            y=member_data["ROC_AUC_SCORE"],
                            mode="lines+markers",
                            marker=dict(color=color_map[member_function]),
                            name=f"{member_function} ({dataset_filter})" if idx == legend_chart_idx else None,
                            showlegend=(idx == legend_chart_idx),
                            text=member_data["MemberFunction"],
                            hovertemplate=(
                                "<b>Dataset:</b> " + dataset_filter + "<br>"
                                "<b>MemberFunction:</b> %{text}<br>"
                                "<b>Length:</b> %{x}<br>"
                                "<b>ROC AUC Score:</b> %{y}<extra></extra>"
                            )
                        ),
                        row=idx // cols + 1,
                        col=idx % cols + 1
                    )
            else:
                # Leeren Plot hinzufügen, wenn keine Daten vorhanden sind
                fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="No Data"), row=idx // cols + 1, col=idx % cols + 1)

        # Layout anpassen
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Scatter Plots by LossFunction",
            title_x=0.5
        )

        fig.update_xaxes(title_text="Trace Length")
        fig.update_yaxes(title_text="AUC ROC")

        # Plot anzeigen
        st.plotly_chart(fig)
    else:
        st.error("Die Datei 'combined_results.csv' wurde nicht gefunden.")

# ROC AUC Analysis
with st.expander("ROC AUC Analysis based on Length with varying parameters"):
    combined_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_results_lr_batch_epoch.csv')
    if os.path.exists(combined_results_path):
        combined_results = pd.read_csv(combined_results_path)
    else:
        st.error("Die Datei 'combined_results_lr_batch_epoch.csv' wurde nicht gefunden.")

    # Heatmap erstellen
    heatmap_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agg_combined_results_lr_batch_epoch.csv')
    if os.path.exists(heatmap_results_path):
        heatmap_results = pd.read_csv(heatmap_results_path)

        # Heatmap-Daten pivotieren
        heatmap_data = heatmap_results.pivot_table(
            index="MemberFunction",
            columns="LearningRate",
            values="ROC_AUC_SCOREweighted",
            aggfunc='mean',
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Heatmap of ROC AUC Scores by MemberFunction and LearningRate")
        ax.set_xlabel("LearningRate")
        ax.set_ylabel("MemberFunction")

        st.pyplot(fig)
    else:
        st.error("Die Datei 'agg_combined_results_lr_batch_epoch.csv' wurde nicht gefunden.")

    # Filtern für BatchSize 256 und EpochSize 100
    filtered_results = combined_results[
        (combined_results['BatchSize'] == 256) & (combined_results['EpochSize'] == 100)]

    # Variierende LearningRates analysieren für MemberFunctions
    unique_member_functions = filtered_results['MemberFunction'].unique()
    unique_learning_rates = filtered_results['LearningRate'].unique()

    # Farbpalette für LearningRates erstellen
    color_map = {lr: color for lr, color in zip(unique_learning_rates, px.colors.qualitative.Plotly)}

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        f"MemberFunction: {mf}" for mf in unique_member_functions[:3]
    ], shared_xaxes=True, shared_yaxes=True)

    legend_chart_idx = None
    max_legend_entries = 0

    # Bestimmen des Diagramms mit den meisten Datenreihen für die Legende
    for idx, member_function in enumerate(unique_member_functions[:3]):
        subset = filtered_results[filtered_results['MemberFunction'] == member_function]
        if not subset.empty:
            legend_entry_count = subset['LearningRate'].nunique()
            if legend_entry_count > max_legend_entries:
                max_legend_entries = legend_entry_count
                legend_chart_idx = idx

    for i, member_function in enumerate(unique_member_functions[:3]):
        subset = filtered_results[filtered_results['MemberFunction'] == member_function]

        for learning_rate in unique_learning_rates:
            lr_subset = subset[subset['LearningRate'] == learning_rate]

            if not lr_subset.empty:
                row = (i // 2) + 1
                col = (i % 2) + 1

                fig.add_trace(
                    go.Scatter(
                        x=lr_subset['LENGHT'],
                        y=lr_subset['ROC_AUC_SCORE'],
                        mode='lines+markers',
                        name=f"LR: {learning_rate}" if i == legend_chart_idx else None,
                        # Legende nur im relevanten Plot
                        showlegend=(i == legend_chart_idx),
                        line=dict(color=color_map[learning_rate]),
                        hovertemplate=(
                            f"<b>Length:</b> %{{x}}<br>"
                            f"<b>ROC AUC Score:</b> %{{y}}<br>"
                            f"<b>LearningRate:</b> {learning_rate}<extra></extra>"
                        )
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        title="ROC AUC Analysis for Varying LearningRates (MemberFunctions Split)",
        height=800,
        width=800,
        xaxis_title="Length",
        yaxis_title="ROC AUC Score",
        showlegend=True,
        legend_title="Learning Rates"
    )

    st.plotly_chart(fig)

    # Variierende BatchSize und EpochSize analysieren
    params = ["BatchSize", "EpochSize"]

    for idx, param in enumerate(params):
        constant_params = [p for p in params if p != param]

        # Filtern von Datensätzen, bei denen nur der aktuelle Parameter variiert
        unique_sets = combined_results.groupby(constant_params).filter(
            lambda g: len(g[param].unique()) > 1 and all(len(g[p].unique()) == 1 for p in constant_params)
        )

        if not unique_sets.empty:
            const_values = {
                constant_params[0]: unique_sets[constant_params[0]].unique()[0]
            }
            st.subheader(f"ROC AUC Analysis for varying {param}")
            st.text(f"{constant_params[0]}: {const_values[constant_params[0]]}")

            fig = go.Figure()

            for value in unique_sets[param].unique():
                subset = unique_sets[unique_sets[param] == value]
                if not subset.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=subset["LENGHT"],
                            y=subset["ROC_AUC_SCORE"],
                            mode="lines+markers",
                            name=f"{param}: {value}",
                            hovertemplate=(
                                f"<b>Length:</b> %{{x}}<br>"
                                f"<b>ROC AUC Score:</b> %{{y}}<br>"
                                f"<b>{param}:</b> {value}<extra></extra>"
                            )
                        )
                    )

            fig.update_layout(
                title=f"ROC AUC Score vs Length ({param} varying)",
                xaxis_title="Length",
                yaxis_title="ROC AUC Score",
                height=600,
                showlegend=True
            )

            st.plotly_chart(fig)
        else:
            st.warning(f"No data found for varying {param} with fixed other parameters.")

# Datei-Pfade
agg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agg_combined_results_lr_batch.csv')
combined_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined_results_lr_batch.csv')

# Sicherstellen, dass die Dateien existieren
if not os.path.exists(agg_file):
    st.error(f"Die Datei '{agg_file}' wurde nicht gefunden.")
if not os.path.exists(combined_file):
    st.error(f"Die Datei '{combined_file}' wurde nicht gefunden.")

# Daten laden
agg_data = pd.read_csv(agg_file)
combined_data = pd.read_csv(combined_file)

with st.expander("Heatmap and Marker-Line Graphs for ROC AUC Analysis"):
    # Heatmap erstellen
    # Daten pivotieren
    heatmap_data = agg_data.pivot_table(
        index="BatchSize",
        columns="LearningRate",
        values="ROC_AUC_SCOREweighted",
        aggfunc='mean',
    )

    # Einträge gleichen Werte von DatasetName, MemberFunction, LossFunction extrahieren
    unique_values = agg_data[['DatasetName', 'MemberFunction', 'LossFunction']].drop_duplicates()
    description = ", ".join(
        [f"{col}: {', '.join(unique_values[col].unique())}" for col in unique_values]
    )
    st.text(f"Filter: {description}")

    # Heatmap plotten
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("ROC AUC Heatmap")
    ax.set_xlabel("LearningRate")
    ax.set_ylabel("BatchSize")
    st.pyplot(fig)

    # Farbzuweisungen aus der Plotly-Palette
    batch_size_colors = {bs: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, bs in enumerate(combined_data['BatchSize'].unique())}
    learning_rate_colors = {lr: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, lr in enumerate(combined_data['LearningRate'].unique())}

    # Bestimmen des Diagramms mit den meisten Datenreihen für die Legende (BatchSize)
    unique_batch_sizes = combined_data['BatchSize'].unique()
    max_entries_bs = 0
    legend_bs_idx = None

    for idx, bs in enumerate(unique_batch_sizes[:4]):
        subset = combined_data[combined_data['BatchSize'] == bs]
        if not subset.empty:
            entry_count = subset['LearningRate'].nunique()
            if entry_count > max_entries_bs:
                max_entries_bs = entry_count
                legend_bs_idx = idx

    # BatchSize Graphen in 2x2 angeordnet, gefiltert nach BatchSize
    rows, cols = 2, 2
    lr_fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"BatchSize: {bs}" for bs in unique_batch_sizes[:rows * cols]])

    for i, bs in enumerate(unique_batch_sizes[:rows * cols]):
        subset = combined_data[combined_data['BatchSize'] == bs]
        for lr in subset['LearningRate'].unique():
            lr_subset = subset[subset['LearningRate'] == lr]
            row = (i // cols) + 1
            col = (i % cols) + 1
            lr_fig.add_trace(go.Scatter(
                x=lr_subset['LENGHT'],
                y=lr_subset['ROC_AUC_SCORE'],
                mode='lines+markers',
                name=f"LR: {lr}" if i == legend_bs_idx else None,
                showlegend=(i == legend_bs_idx),
                line=dict(color=learning_rate_colors[lr]),
                hovertemplate=(
                    f"<b>Length:</b> %{{x}}<br>"
                    f"<b>ROC AUC Score:</b> %{{y}}<br>"
                    f"<b>LearningRate:</b> {lr}<extra></extra>"
                )
            ), row=row, col=col)

    lr_fig.update_layout(
        title="ROC AUC Scores vs Length (Filtered by BatchSize)",
        xaxis_title="Length",
        yaxis_title="ROC AUC Score",
        showlegend=True,
        height=800,
        width=800
    )
    st.plotly_chart(lr_fig)

    # Bestimmen des Diagramms mit den meisten Datenreihen für die Legende (LearningRate)
    unique_learning_rates = combined_data['LearningRate'].unique()
    max_entries_lr = 0
    legend_lr_idx = None

    for idx, lr in enumerate(unique_learning_rates[:6]):
        subset = combined_data[combined_data['LearningRate'] == lr]
        if not subset.empty:
            entry_count = subset['BatchSize'].nunique()
            if entry_count > max_entries_lr:
                max_entries_lr = entry_count
                legend_lr_idx = idx

    # LearningRate Graphen in 2x3 angeordnet, gefiltert nach LearningRate
    rows, cols = 2, 3
    bs_fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"LearningRate: {lr}" for lr in unique_learning_rates[:rows * cols]])

    for i, lr in enumerate(unique_learning_rates[:rows * cols]):
        subset = combined_data[combined_data['LearningRate'] == lr]
        for bs in subset['BatchSize'].unique():
            bs_subset = subset[subset['BatchSize'] == bs]
            row = (i // cols) + 1
            col = (i % cols) + 1
            bs_fig.add_trace(go.Scatter(
                x=bs_subset['LENGHT'],
                y=bs_subset['ROC_AUC_SCORE'],
                mode='lines+markers',
                name=f"BatchSize: {bs}" if i == legend_lr_idx else None,
                showlegend=(i == legend_lr_idx),
                line=dict(color=batch_size_colors[bs]),
                hovertemplate=(
                    f"<b>Length:</b> %{{x}}<br>"
                    f"<b>ROC AUC Score:</b> %{{y}}<br>"
                    f"<b>BatchSize:</b> {bs}<extra></extra>"
                )
            ), row=row, col=col)

    bs_fig.update_layout(
        title="ROC AUC Scores vs Length (Filtered by LearningRate)",
        xaxis_title="Length",
        yaxis_title="ROC AUC Score",
        showlegend=True,
        height=800,
        width=1000
    )
    st.plotly_chart(bs_fig)
