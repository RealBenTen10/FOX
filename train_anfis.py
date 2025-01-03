import itertools
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from membership import MfsType, make_anfis
from experimental import train_anfis_cat
import load_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
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
encoding_type = encoding_types[3]

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
    'production',
    'prepaid_travelcost'
]
# pick the dataset of your choice
dataset_name = dataset_names[11]

# the parameters have the same order as the dataset_names
# so params[0] has the parameters lr and batch_size of sepsis_cases_1
# params_list[X][0] is learning_rate
# params_list[X][1] is batch_size
params_list = [
    [8.878741925407075e-05, 256],
    [0.00992592926325457, 256],
    [0.0005634190655247415, 128],
    [0.00015056471839384854, 256],
    [0.000999999999999913, 1024],
    [0.00992592926325457, 256],
    [0.008766616117947781, 128],
    [0.003375590702146598, 512],
    [0.004182032486569149, 512],
    [8.878741925407075e-05, 256],
    [0.0005634190655247415, 128]
]
params = params_list[0]

# Membership function types
mfs_types = [
    MfsType.Bell,
    MfsType.DSigmoid,
    MfsType.Gauss,
    MfsType.Sigmoid,
    MfsType.Triangular
]
mfs_type = mfs_types[2]

# Set some parameters
epochs = 10            # Set the number of epochs
sigmoid = False         # use sigmoid instead of softmax
train_size = 0.8        # Set the split size - 0.8 = 20% validation and 80% train
random_state = 69       # Set the random state
learning_rate = params[0]   # Set the learning rate
batch_size = params[1]        # Set the batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set the device - don't change this
num_mfs = 3             # Set number of membership functions? or number of possible values (low, med, high)
                        #   Answer: Fuzzification creates a probability distribution how likely each variable belongs to e certain fuzzy term, in this case LOW, MEDIUM, HIGH.
                        #           For each fuzzy variable (feature vector item) and each fuzzy term (LOW, MEDIUM, HIGH or GOOD, NEUTRAL, BAD) there is a new membership function with its own learnable parameters
                        #           So num_terms == num_mfs has to be true
                        #           So this parameters defines how many terms you want to differentiate into and you have to create a MF for every term


# used for torch - otherwise random state could be different each time
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
    """Convert categorical data into boolean representation"""
    unique_classes = np.unique(data)
    encoded = np.array([[1 if val == cls else 0 for cls in unique_classes] for val in data])
    return torch.tensor(encoded, dtype=torch.float)


def label_encoding(data):
    """Convert categories to integer labels"""
    labels = {val: idx for idx, val in enumerate(np.unique(data))}
    encoded = np.array([labels[val] for val in data])
    return torch.tensor(encoded, dtype=torch.long)


def index_encoding(data):
    """Convert categories into their index position"""
    unique_classes = np.unique(data)
    indices = {val: idx for idx, val in enumerate(unique_classes)}
    encoded = np.array([indices[val] for val in data])
    return torch.tensor(encoded, dtype=torch.long)


def one_hot_encoding(data, num_categories, dtype=torch.float):
    """Convert data into one-hot representation"""
    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype) \
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y


def get_data(dataset, batch_size, columns_sel):
    dataframe = pd.read_csv(f'dataset/{dataset}/{dataset}_train.csv', header=0, sep=',')
    columns_sel.append('Classification')
    dataframe = dataframe[columns_sel]

    array = dataframe.values
    d_data = array[:, :-1]
    d_target = array[:, -1]

    # split train/val - test dataset is separate
    X_train, X_val, y_train, y_val = train_test_split(d_data, d_target, test_size=(1-train_size),
                                                      stratify=d_target, random_state=random_state, shuffle=True)

    # Encode targets based on the chosen encoding type
    # are the encoded values used???
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

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    x = torch.Tensor(X_train)
    y = torch.Tensor(y_train)
    td = TensorDataset(x, y)

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(val_dataset, batch_size=batch_size),
            DataLoader(td, batch_size=batch_size, shuffle=False),  # return td with encoded values
            columns_sel)


def train(dataset, learning_rate, batch_size, columns_sel, encoding_type, sigmoid, mfs_type):
    train_data, val_data, x, columns_sel = get_data(dataset, batch_size, columns_sel)
    # y_train holds the encoded values... but is not used -_-
    x_train, y_train = x.dataset.tensors
    # Create ANFIS model
    model = make_anfis(x_train, device, num_mfs=3, num_out=2, hybrid=False, mfs_type=mfs_type)
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # (Re-)train ANFIS model using optimizer Adam
    sigmoid_str = "sigmoid" if sigmoid else "softmax"
    log_file = open(f"train_logs/train_log2_{dataset_name}_{encoding_type}_{sigmoid_str}_{str(mfs_type)}.txt", "w")
    model, score = train_anfis_cat(model, train_data, val_data, optimizer, epochs, encoding_type, sigmoid, device, log_file=log_file)
    log_file.close()
    # Save the trained model
    sigmoid_str = "sigmoid" if sigmoid else "softmax"
    model_name = f"model2_{dataset_name}_{encoding_type}_{sigmoid_str}_{str(mfs_type)}"
    torch.save(model, 'models/' + model_name + '.h5')
    # torch.save(model, 'streamlit_fox/models/' + model_name + '.h5')
    # Get metrics for the model
    load_model.metrics(dataset, columns_sel, encoding_type, sigmoid, mfs_type, device)
    return model


# Get features for model training and streamlit view
def get_columns_sel(dataset_name):
    if dataset_name == 'sepsis_cases_1':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'std_Leucocytes', 'std_CRP']
    elif dataset_name == 'sepsis_cases_2':
        columns_sel = ['Diagnose', 'mean_open_cases', 'mean_hour', 'DisfuncOrg']
    elif dataset_name == 'sepsis_cases_4':
        columns_sel = ['Diagnose', 'mean_open_cases', 'Age', 'org:group_E', 'std_CRP', 'DiagnosticECG']
    elif dataset_name == 'bpic2011_f1':
        columns_sel = ['Diagnosis Treatment Combination ID', 'mean_open_cases', 'Diagnosis', 'Activity code_376400.0']
    elif dataset_name == 'bpic2011_f2':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'Diagnosis code', 'mean_open_cases',
                       'Activity code_376400.0', 'Age', 'Producer code_CHE1']
    elif dataset_name == 'bpic2011_f3':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Diagnosis', 'mean_open_cases', 'Diagnosis code',
                       'std_event_nr', 'mean_event_nr']
    elif dataset_name == 'bpic2011_f4':
        columns_sel = ['Diagnosis Treatment Combination ID', 'Treatment code']
    elif dataset_name == 'bpic2012_accepted':
        columns_sel = ['AMOUNT_REQ', 'Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE',
                       'Activity_W_Valideren aanvraag-START']
    elif dataset_name == 'bpic2012_declined':
        columns_sel = ['AMOUNT_REQ', 'Activity_A_PARTLYSUBMITTED-COMPLETE', 'Activity_A_PREACCEPTED-COMPLETE',
                       'Activity_A_DECLINED-COMPLETE', 'Activity_W_Completeren aanvraag-SCHEDULE', 'mean_open_cases']
    elif dataset_name == 'bpic2012_cancelled':
        columns_sel = ['Activity_O_SENT_BACK-COMPLETE', 'Activity_W_Valideren aanvraag-SCHEDULE',
                       'Activity_W_Valideren aanvraag-START', 'AMOUNT_REQ', 'Activity_W_Valideren aanvraag-COMPLETE',
                       'Activity_A_CANCELLED-COMPLETE']
    elif dataset_name == 'production':
        columns_sel = ['Work_Order_Qty', 'Activity_Turning & Milling - Machine 4', 'Resource_ID0998', 'Resource_ID4794',
                       'Resource.1_Machine 4 - Turning & Milling']
    elif dataset_name == 'prepaid_travelcost':
        columns_sel = ['RequestedAmount', 'RequestForPaymentSUBMITTEDbyEMPLOYEE',
                       'RequestForPaymentAPPROVEDbyADMINISTRATION', 'RequestForPaymentAPPROVEDbyBUDGETOWNER',
                       'RequestForPaymentFINAL_APPROVEDbySUPERVISOR', 'RequestPayment']
    else:
        return []
    return columns_sel

# params_index = dataset_names.index(dataset_name)
# learning_rate = params_list[params_index][0]
# batch_size = params_list[params_index][1]

# select columns
# columns_sel = get_columns_sel(dataset_name)
# # number of features selected (leads to the selection of all?)
# n_features = len(columns_sel)
# # train model here
# model = train(dataset_name, learning_rate, batch_size, columns_sel[:n_features], encoding_type, sigmoid, mfs_type)
# train models there
model = train(dataset_name, learning_rate, batch_size, get_columns_sel(dataset_name), encoding_type, sigmoid, mfs_type)

succeeded = 0
configurations = list(itertools.product(['sepsis_cases_1'], encoding_types, [True, False], mfs_types))
if False:
    for (i, (dataset_name, encoding_type, sigmoid, mfs_type)) in enumerate(configurations, start=1):
        # Reseed RNGs
        np.random.seed(seed)
        torch.manual_seed(seed)
        params_index = dataset_names.index(dataset_name)
        learning_rate = params_list[params_index][0]
        batch_size = params_list[params_index][1]

        # select columns
        columns_sel = get_columns_sel(dataset_name)
        # number of features selected (leads to the selection of all?)
        n_features = len(columns_sel)

        # train model for specific config
        try:
            print(f"Configuration {i}/{len(configurations)}:", encoding_type, sigmoid, mfs_type, "for ", dataset_name, " - \n", end="")
            start = time.perf_counter()
            model = train(dataset_name, learning_rate, batch_size, columns_sel[:n_features], encoding_type, sigmoid, mfs_type)
            end = time.perf_counter()
            # print("\033[F\033[F\033[F", end="")
            print("\r Succeeded in ", f"{end - start}s", flush=True)
            succeeded += 1
        except Exception:
            print("\r Failed", flush=True)
            pass
print(succeeded, " of ", len(configurations), " succeeded")
