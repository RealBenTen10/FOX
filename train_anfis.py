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
import torch

# Set the encoding type here: choose 'boolean', 'label', 'index', or 'one_hot'
# Available encodings:
encoding_types = [
    'boolean',
    'label',
    'index',
    'one_hot'
]
# pick the encoding of your choice
encoding_type = encoding_types[1]

# Change the dataset_name to create a new trained model (.h5 file)
# Available datasets:
dataset_names = [
    'sepsis_cases_1',
    'sepsis_cases_2',
    'sepsis_cases_4',
    'bpic2011_f1',
    'bpic2011_f2',
    'bpic2011_f3',
    'bpic2011_f4',
    'bpic2012_accepted',
    'bpic2012_declined',
    'bpic2012_cancelled',
    'production'
]
# pick the dataset of your choice
dataset_name = dataset_names[0]

# Set some parameters
# Set the number of epochs
epochs = 10
sigmoid = True  # use sigmoid instead of softmax
test_size = 0.2
random_state = 69
batch_size = 250
learning_rate = 0.001

# not used?
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

def one_hot_encoding(data, num_categories, dtype=torch.float):
    """Convert data into one-hot representation."""
    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype) \
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y

def get_data(dataset, n_feature, batch_size, columns_sel):
    dataframe = pd.read_csv(f'dataset/{dataset}/{dataset}_train.csv', header=0, sep=',')
    columns_sel.append('Classification')
    dataframe = dataframe[columns_sel]

    array = dataframe.values
    d_data = array[:, :-1]
    d_target = array[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(
        d_data, d_target, test_size=test_size, stratify=d_target, random_state=random_state, shuffle=True
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
    print("Encoding_type: " + encoding_type)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    x = torch.Tensor(X_train)
    y = y_train_encoded
    td = TensorDataset(x, y)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False), DataLoader(val_dataset, batch_size=batch_size), DataLoader(td, batch_size=batch_size, shuffle=False), columns_sel

def train(dataset, n_feature, learning_rate, bs, columns_sel):
    train_data, val_data, x, columns_sel = get_data(dataset, n_feature, bs, columns_sel)
    x_train, y_train = x.dataset.tensors
    model = make_anfis(x_train, num_mfs=3, num_out=2, hybrid=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, score = experimental.train_anfis_cat(model, train_data, val_data, optimizer,epochs, encoding_type, sigmoid)
    torch.save(model, 'models/model_' + dataset + '.h5')
    torch.save(model, 'streamlit_fox/models/model_' + dataset + '.h5')
    load_model.metrics(dataset, columns_sel)
    return model
def opt(dataset, n_feature, learning_rate, bs, file_name, columns_sel):
    train_data, val_data, x, columns_sel = get_data(dataset, n_feature, bs, columns_sel)
    x_train, y_train = x.dataset.tensors

    model = make_anfis(x_train, num_mfs=3, num_out=2, hybrid=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, scores = experimental.train_anfis_cat(model, train_data, val_data, optimizer,epochs, encoding_type, sigmoid)
    return model, scores

dataset_name = 'sepsis_cases_1'

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
# model = train(dataset_name, n_features, params.get('lr'), params.get('batch_size'), columns_sel[:n_features])
model = train(dataset_name, n_features, learning_rate, batch_size, columns_sel[:n_features])