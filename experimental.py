import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np

# not used?
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float

# One_hot encoding
def make_one_hot(data, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Properly clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype).scatter(1, cats, 1)
    y.requires_grad = True
    return y

# Label Encoding
def make_label_encoding(data, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

# Index Encoding
def make_index_encoding(data, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

# Boolean Encoding
def make_boolean_encoding(data, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype).scatter(1, cats, 1)
    y = y.to(dtype)  # Ensure the tensor is of floating-point type
    return y

# Get encoding function based on type
def get_encoding_function(encoding_type):
    if encoding_type == "one_hot":
        return make_one_hot
    elif encoding_type == "label":
        return make_label_encoding
    elif encoding_type == "index":
        return make_index_encoding
    elif encoding_type == "boolean":
        return make_boolean_encoding
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

# Training with dynamic encoding selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def multi_acc(y_pred, y_test, sigmoid):
    # Sigmoid function instead of softmax
    if sigmoid:
        y_pred_sigmoid = (torch.sigmoid(y_pred) > 0.5).long()
        _, y_pred_tags = torch.max(y_pred_sigmoid, dim=1)
    else:
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_anfis_cat(model, train_loader, val_loader, optimizer, EPOCHS, encoding_type, sigmoid):
    print(device)
    print("Begin training.")
    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = np.inf
    best_epoch = 0
    encoding_function = get_encoding_function(encoding_type)

    for e in tqdm(range(1, EPOCHS + 1)):
        train_epoch_loss = 0
        train_epoch_acc = 0

        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            encoded_labels = encoding_function(y_train_batch, num_categories=2)
            train_loss = criterion(y_train_pred, encoded_labels)
            train_acc = multi_acc(y_train_pred, y_train_batch, sigmoid)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.fit_coeff(
                X_train_batch.float(),
                encoding_function(y_train_batch, num_categories=2)
            )
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)
                encoded_labels = encoding_function(y_val_batch, num_categories=2)
                val_loss = criterion(y_val_pred, encoded_labels)
                val_acc = multi_acc(y_val_pred, y_val_batch, sigmoid)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        if (val_epoch_loss / len(val_loader)) < best_loss:
            best_loss = (val_epoch_loss / len(val_loader))
            best_epoch = e
            best_model = model
        if e - best_epoch > 10:
            print(best_epoch)
            break
        print(
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.8f} | Val Loss: {val_epoch_loss / len(val_loader):.8f} | Train Acc: {train_epoch_acc / len(train_loader):.8f}| Val Acc: {val_epoch_acc / len(val_loader):.8f}')

    return best_model, loss_stats['val']

