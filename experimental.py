import torch
import torch.nn.functional as f
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# not used?
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float

# One_hot encoding


def make_one_hot(data, device, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Properly clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype, device=device).scatter(1, cats, 1)
    y.requires_grad = True
    return y

# Label Encoding


def make_label_encoding(data, device, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

# Index Encoding

def make_index_encoding(data, device, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

# Boolean Encoding


def make_boolean_encoding(data, device, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype, device=device).scatter(1, cats, 1)
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
    # add more if wanted
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

# Choose device - don't change this


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_loss_function(bce_with_logits, l1_loss, smooth_l1):
    if bce_with_logits:
        return torch.nn.BCEWithLogitsLoss()
    elif l1_loss:
        return torch.nn.L1Loss()
    elif smooth_l1:
        return torch.nn.SmoothL1Loss()
    else:
        return torch.nn.CrossEntropyLoss()

# Get Accuracy
# Here the predicted values are tested against the true values
# For sepsis_cases_1 the model always predicts 0 which results in the same accuracy each epoch


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


def train_anfis_cat(model, train_loader, val_loader, optimizer, EPOCHS, encoding_type, sigmoid, device):
    print(device)
    print("Begin training.")
    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    precision_stats = {'train': [], "val": []}
    recall_stats = {'train': [], "val": []}
    f1_stats = {'train': [], "val": []}
    auc_stats = {'train': [], "val": []}
    # Get loss function - if all are false, the default is CrossEntropyLoss
    criterion = get_loss_function(False, False, False)
    best_loss = np.inf
    best_epoch = 0
    best_model = 0
    encoding_function = get_encoding_function(encoding_type)
    X_train_batch, y_train_batch = 0, 0

    for e in tqdm(range(1, EPOCHS + 1)):
        train_epoch_loss = 0
        train_epoch_acc = 0
        train_epoch_precision = 0
        train_epoch_recall = 0
        train_epoch_f1 = 0
        train_epoch_auc = 0

        # Training loop
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            encoded_labels = encoding_function(y_train_batch, device, num_categories=2)
            train_loss = criterion(y_train_pred, encoded_labels)
            train_acc = multi_acc(y_train_pred, y_train_batch, sigmoid)

            # Predictions and Metrics
            y_pred_prob = torch.sigmoid(y_train_pred) if sigmoid else f.softmax(y_train_pred, dim=1)
            y_pred_binary = (y_pred_prob[:, 1] > 0.5).cpu().numpy()
            # true label?
            y_true = y_train_batch.cpu().numpy()

            train_precision = precision_score(y_true, y_pred_binary, zero_division=0)
            train_recall = recall_score(y_true, y_pred_binary, zero_division=0)
            train_f1 = f1_score(y_true, y_pred_binary)
            train_auc = roc_auc_score(y_true, y_pred_prob[:, 1].detach().cpu().numpy())

            # Backpropagation
            train_loss.backward()
            optimizer.step()
            # Metric calculation
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            train_epoch_precision += train_precision
            train_epoch_recall += train_recall
            train_epoch_f1 += train_f1
            train_epoch_auc += train_auc

        # Validation loop
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            val_epoch_precision = 0
            val_epoch_recall = 0
            val_epoch_f1 = 0
            val_epoch_auc = 0

            model.fit_coeff(X_train_batch.float(), encoding_function(y_train_batch, device, num_categories=2))

            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)
                encoded_labels = encoding_function(y_val_batch, device, num_categories=2)
                val_loss = criterion(y_val_pred, encoded_labels)
                val_acc = multi_acc(y_val_pred, y_val_batch, sigmoid)

                # Predictions and Metrics
                y_val_prob = torch.sigmoid(y_val_pred) if sigmoid else f.softmax(y_val_pred, dim=1)
                y_val_binary = (y_val_prob[:, 1] > 0.5).cpu().numpy()
                # true label?
                y_val_true = y_val_batch.cpu().numpy()

                val_precision = precision_score(y_val_true, y_val_binary, zero_division=0)
                val_recall = recall_score(y_val_true, y_val_binary, zero_division=0)
                val_f1 = f1_score(y_val_true, y_val_binary)
                val_auc = roc_auc_score(y_val_true, y_val_prob[:, 1].cpu().numpy())

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                val_epoch_precision += val_precision
                val_epoch_recall += val_recall
                val_epoch_f1 += val_f1
                val_epoch_auc += val_auc

        # Logging metrics
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))
        precision_stats['train'].append(train_epoch_precision / len(train_loader))
        precision_stats['val'].append(val_epoch_precision / len(val_loader))
        recall_stats['train'].append(train_epoch_recall / len(train_loader))
        recall_stats['val'].append(val_epoch_recall / len(val_loader))
        f1_stats['train'].append(train_epoch_f1 / len(train_loader))
        f1_stats['val'].append(val_epoch_f1 / len(val_loader))
        auc_stats['train'].append(train_epoch_auc / len(train_loader))
        auc_stats['val'].append(val_epoch_auc / len(val_loader))

        # Early stopping
        if (val_epoch_loss / len(val_loader)) < best_loss:
            best_loss = (val_epoch_loss / len(val_loader))
            best_epoch = e
            best_model = model
        if e - best_epoch > 10:
            print(best_epoch)
            break

        print(
            f'Epoch {e:03}: | Train Loss: {train_epoch_loss / len(train_loader):.4f} | Val Loss: {val_epoch_loss / len(val_loader):.4f} | '
            f'Train Acc: {train_epoch_acc / len(train_loader):.4f} | Val Acc: {val_epoch_acc / len(val_loader):.4f} | '
            f'Train Prec: {train_epoch_precision / len(train_loader):.4f} | Val Prec: {val_epoch_precision / len(val_loader):.4f} | '
            f'Train Rec: {train_epoch_recall / len(train_loader):.4f} | Val Rec: {val_epoch_recall / len(val_loader):.4f} | '
            f'Train F1: {train_epoch_f1 / len(train_loader):.4f} | Val F1: {val_epoch_f1 / len(val_loader):.4f} | '
            f'Train AUC: {train_epoch_auc / len(train_loader):.4f} | Val AUC: {val_epoch_auc / len(val_loader):.4f}'
        )

    return best_model, loss_stats['val']


