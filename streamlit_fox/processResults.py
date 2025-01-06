import os
import pandas as pd

def extract_information_from_filename(filename):
    """
    Extrahiert den Datensatznamen, die Memberfunction und die LossFunction aus dem Dateinamen.
    """
    base_name = os.path.basename(filename)
    name_parts = base_name.split('_')

    # Extrahieren des Datensatznamens
    dataset_name = "_".join(name_parts[:name_parts.index('one')])

    # Extrahieren der Memberfunction (drittletzter Eintrag)
    member_function = name_parts[-3]

    # Extrahieren der LossFunction (zweitletzter Eintrag)
    loss_function = name_parts[-2]

    return dataset_name, member_function, loss_function

def read_and_split_csv(file_path):
    """
    Liest eine CSV-Datei, die Daten ab einer bestimmten Zeile enthält, und trennt die Hauptdaten von den letzten zwei Zeilen.
    Die Tabelle beginnt mit den Spaltenüberschriften: LENGHT;NUMBEROFSAMPLES;ROC_AUC_SCORE
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Finden der Startzeile für die Tabelle
    for i, line in enumerate(lines):
        if line.strip() == 'LENGHT;NUMBEROFSAMPLES;ROC_AUC_SCORE':
            start_line = i
            break

    # Hauptdaten auslesen
    main_data = pd.read_csv(file_path, sep=';', skiprows=start_line + 1, header=None, names=['LENGHT', 'NUMBEROFSAMPLES', 'ROC_AUC_SCORE'])

    # Entfernen der letzten zwei Zeilen (spezielle Daten)
    agg_data = main_data.iloc[-2:].copy()
    main_data = main_data.iloc[:-2]

    return main_data, agg_data

def combine_csv_to_dataframe(folder_path):
    """
    Liest alle .csv-Dateien in einem Ordner und kombiniert sie zu einem DataFrame.
    Fügt zusätzliche Spalten hinzu: DatasetName, MemberFunction, LossFunction.
    """
    combined_df = pd.DataFrame()
    agg_combined_results = pd.DataFrame()

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)

            # Hauptdaten und aggregierte Daten aus der CSV-Datei extrahieren
            main_data, agg_data = read_and_split_csv(file_path)

            # Extrahieren der Informationen aus dem Dateinamen
            dataset_name, member_function, loss_function = extract_information_from_filename(file)

            # Hinzufügen der zusätzlichen Spalten zu den Hauptdaten
            main_data['DatasetName'] = dataset_name
            main_data['MemberFunction'] = member_function
            main_data['LossFunction'] = loss_function

            agg_data_temp = pd.DataFrame([{
                'ROC_AUC_SCOREweighted':
                    agg_data.loc[agg_data['LENGHT'] == 'ROC_AUC_SCORE weighted', 'NUMBEROFSAMPLES'].iloc[0],
                'F1_SCOREweighted': agg_data.loc[agg_data['LENGHT'] == 'F1_SCORE weighted', 'NUMBEROFSAMPLES'].iloc[0],
                'DatasetName': dataset_name,
                'MemberFunction': member_function,
                'LossFunction': loss_function
            }])

            # Kombinieren der Daten
            combined_df = pd.concat([combined_df, main_data], ignore_index=True)
            agg_combined_results = pd.concat([agg_combined_results, agg_data_temp], ignore_index=True)

    return combined_df, agg_combined_results

def postprocess_dataframes(combined_df, agg_combined_results):
    """
    Führt ein Postprocessing durch, um die Realistik der ROC_AUC_SCORE Werte zu bestimmen.
    """
    # Konvertieren der ROC_AUC_SCORE Spalte in numerische Werte
    combined_df['ROC_AUC_SCORE'] = pd.to_numeric(combined_df['ROC_AUC_SCORE'], errors='coerce')

    return combined_df, agg_combined_results


# Ordnerpfad, der die CSV-Dateien enthält
folder_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resultsNew'))

# Kombinieren der CSV-Dateien in ein DataFrame
combined_results, agg_combined_results = combine_csv_to_dataframe(folder_path)

# Postprocessing der DataFrames
combined_results, agg_combined_results = postprocess_dataframes(combined_results, agg_combined_results)

# Ergebnis anzeigen oder speichern
combined_results.to_csv('./combined_resultsNew.csv', index=False)
agg_combined_results.to_csv('./agg_combined_resultsNew.csv', index=False)

print("Kombinierte Datei wurde gespeichert als 'combined_resultsNew.csv'")
print("Aggregierte Datei wurde gespeichert als 'agg_combined_resultsNew.csv'")
