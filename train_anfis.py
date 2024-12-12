import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from membership import make_anfis
import experimental
import load_model
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import pickle as pk
seed = 123
np.random.seed(seed)
import torch
torch.manual_seed(seed)

# Set the encoding type here: choose 'boolean', 'label', 'index', or 'one_hot'
encoding_type = 'one_hot'

# Change the dataset_name to create a new trained model (.h5 file)
dataset_name = 'sepsis_cases_1'
# Available datasets:
# 'sepsis_cases_1'
# 'sepsis_cases_2'
# 'sepsis_cases_4'
# 'bpic2011_f1'
# 'bpic2011_f2'
# 'bpic2011_f3'
# 'bpic2011_f4'
# 'bpic2012_accepted'
# 'bpic2012_declined'
# 'bpic2012_cancelled'
# 'production'


seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

def boolean_encoding(data):
    """Convert categorical data into boolean representation."""
    unique_classes = np.unique(data)
    encoded = np.array([[1 if val == cls else 0 for cls in unique_classes] for val in data])
    return torch.tensor(encoded, dtype=torch.float)

def label_encoding(data):
    """Convert categories to integer labels."""
    labels = {val: idx for idx, val in enumerate(np.unique(data))}
    encoded = np.array([labels[val] for val in data])
    return torch.tensor(encoded, dtype=torch.long)

def index_encoding(data):
    """Convert categories into their index position."""
    unique_classes = np.unique(data)
    indices = {val: idx for idx, val in enumerate(unique_classes)}
    encoded = np.array([indices[val] for val in data])
    return torch.tensor(encoded, dtype=torch.long)

def one_hot_encoding(data, num_categories):
    """Convert data into one-hot representation."""
    data_tensor = torch.tensor(data, dtype=torch.long).unsqueeze(1)
    one_hot = torch.zeros(len(data), num_categories, dtype=torch.float).scatter(1, data_tensor, 1)
    return one_hot

def get_data(dataset, n_feature, batch_size, columns_sel):
    dataframe = pd.read_csv(f'dataset/{dataset}/{dataset}_train.csv', header=0, sep=',')
    columns_sel.append('Classification')
    dataframe = dataframe[columns_sel]

    array = dataframe.values
    d_data = array[:, :-1]
    d_target = array[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(
        d_data, d_target, test_size=0.2, stratify=d_target, random_state=69, shuffle=True
    )

    # Encode targets based on the chosen encoding type
    if encoding_type == 'boolean':
        y_train_encoded = boolean_encoding(y_train)
    elif encoding_type == 'label':
        y_train_encoded = label_encoding(y_train)
    elif encoding_type == 'index':
        y_train_encoded = index_encoding(y_train)
    elif encoding_type == 'one_hot':
        num_categories = len(np.unique(d_target))
        y_train_encoded = one_hot_encoding(y_train, num_categories)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    train_dataset = ClassifierDataset(torch.tensor(X_train, dtype=torch.float), y_train_encoded)
    val_dataset = ClassifierDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.long))

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        columns_sel
    )

def train(dataset, n_feature, learning_rate, bs, columns_sel):
    train_data, val_data, columns_sel = get_data(dataset, n_feature, bs, columns_sel)
    x_train, y_train = next(iter(train_data))

    model = make_anfis(x_train, num_mfs=3, num_out=len(torch.unique(y_train)), hybrid=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, score = experimental.train_anfis_cat(model, train_data, val_data, optimizer, 100)
    torch.save(model, 'streamlit_fox/models/model_' + dataset + '.h5')
    torch.save(model, 'models/model_' + dataset + '.h5')
    load_model.metrics(dataset, columns_sel)
    return model

def opt(dataset, n_feature, learning_rate, bs, file_name, columns_sel):
    train_data, val_data, columns_sel = get_data(dataset, n_feature, bs, columns_sel)
    x_train, y_train = next(iter(train_data))

    model = make_anfis(x_train, num_mfs=3, num_out=len(torch.unique(y_train)), hybrid=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model, scores = experimental.train_anfis_cat(model, train_data, val_data, optimizer, 100)
    return model, scores



if dataset_name == 'sepsis_cases_1':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'std_Leucocytes', 'std_CRP']#sepsis_cases_1
elif dataset_name == 'sepsis_cases_2':
        columns_sel = ['Diagnose', 'mean_open_cases', 'mean_hour', 'DisfuncOrg']#sepsis_cases_2
elif dataset_name == 'sepsis_cases_4':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'org:group_E', 'std_CRP', 'DiagnosticECG']#sepsis_cases_4
elif dataset_name == 'bpic2011_f1':
        columns_sel = ['Diagnosis Treatment Combination ID', 'mean_open_cases', 'Diagnosis', 'Activity code_376400.0']#bpic2011_f1
elif dataset_name == 'bpic2011_f2':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'Diagnosis code', 'mean_open_cases', 'Activity code_376400.0', 'Age', 'Producer code_CHE1']#bpic2011_f2
elif dataset_name == 'bpic2011_f3':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'mean_open_cases', 'Diagnosis code', 'std_event_nr', 'mean_event_nr']#bpic2011_f3
elif dataset_name == 'bpic2011_f4':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Treatment code']#bpic2011_f4
elif dataset_name == 'bpic2012_accepted':
        columns_sel = ['AMOUNT_REQ', 'Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START']#bpic2012_accepted
elif dataset_name == 'bpic2012_declined':
        columns_sel = ['AMOUNT_REQ', 'Activity_A_PARTLYSUBMITTED-COMPLETE', 'Activity_A_PREACCEPTED-COMPLETE', 'Activity_A_DECLINED-COMPLETE', 'Activity_W_Completeren aanvraag-SCHEDULE', 'mean_open_cases'] #bpic2012_declined
elif dataset_name == 'bpic2012_cancelled':
        columns_sel = ['Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE', 'Activity_W_Valideren aanvraag-START', 'AMOUNT_REQ', 'Activity_W_Valideren aanvraag-COMPLETE', 'Activity_A_CANCELLED-COMPLETE']#bpic2012_cancelled
elif dataset_name == 'production':
        columns_sel = ['Work_Order_Qty', 'Activity_Turning & Milling - Machine 4', 'Resource_ID0998', 'Resource_ID4794', 'Resource.1_Machine 4 - Turning & Milling']#production

params = pk.load(open('params/'+dataset_name+".p", "rb"))
n_features = len(columns_sel)
model = train(dataset_name, n_features, params.get('lr'), params.get('batch_size'), columns_sel[:n_features])
