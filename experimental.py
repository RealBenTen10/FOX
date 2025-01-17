import torch
import torch.nn.functional as f
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# used for torch - otherwise random each run
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float
# Choose device - don't change this
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_one_hot(data, device, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Properly clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype, device=device).scatter(1, cats, 1)
    y.requires_grad = True
    return y

def make_label_encoding(data, device, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

def make_index_encoding(data, device, num_categories=None, dtype=torch.float):
    y = data.clone().detach().long()  # Properly clone and detach input tensor
    return y

def make_boolean_encoding(data, device, num_categories, dtype=torch.float):
    num_entries = len(data)
    cats = data.clone().detach().long().unsqueeze(1)  # Clone and detach input tensor
    y = torch.zeros((num_entries, num_categories), dtype=dtype, device=device).scatter(1, cats, 1)
    y = y.to(dtype)  # Ensure the tensor is of floating-point type
    return y

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

def get_loss_function(loss_function):
    if loss_function == "BCE":
        return torch.nn.BCELoss()
    elif loss_function == "L1":
        return torch.nn.L1Loss()
    elif loss_function == "SmoothL1":
        return torch.nn.SmoothL1Loss()
    else:
        return torch.nn.CrossEntropyLoss()


def multi_acc(y_pred, y_test, sigmoid):
    # Sigmoid function added (before only softmax)
    # Get Accuracy
    # Here the predicted values are tested against the true values
    # For sepsis_cases_1 the model always predicts 0 which results in the same accuracy each epoch
    if sigmoid:
        y_pred_sigmoid = (torch.sigmoid(y_pred) > 0.5).long()
        _, y_pred_tags = torch.max(y_pred_sigmoid, dim=1)
    else:
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    # print("y_pred_tags: \n", y_pred_tags)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    return acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_anfis_cat(model, train_loader, val_loader, optimizer, EPOCHS, encoding_type,
                    sigmoid, device, loss_function, log_file=None):
    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    precision_stats = {'train': [], "val": []}
    recall_stats = {'train': [], "val": []}
    f1_stats = {'train': [], "val": []}
    auc_stats = {'train': [], "val": []}
    # Get loss function - if all are false, the default is CrossEntropyLoss
    criterion = get_loss_function(loss_function)
    best_loss = np.inf
    best_epoch = 0
    best_model = 0
    encoding_function = get_encoding_function(encoding_type)
    X_train_batch, y_train_batch = 0, 0

    for e in range(1, EPOCHS + 1):
        print(f"\rEpoch: ", e, "/", EPOCHS, end="", flush=True)
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

            # we need y_train_pred_loss for train_loss and y_train_pred for train_acc
            if loss_function == "SmoothL1" or loss_function == "L1" or loss_function == "BCE":
                y_train_pred_loss = y_train_pred[:, 0].requires_grad_(True).to(device)
                train_loss = criterion(y_train_pred_loss, y_train_batch.float())
            else:
                train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch, sigmoid)

            # Predictions and Metrics
            y_pred_prob = torch.sigmoid(y_train_pred) if sigmoid else f.softmax(y_train_pred, dim=1)
            y_pred_binary = (y_pred_prob[:, 1] > 0.5).cpu().numpy()
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
            val_epoch_loss = val_epoch_acc = val_epoch_precision = val_epoch_recall = val_epoch_f1 = val_epoch_auc = 0

            # Decide if you want to use the encoding or not - the values do not change...
            model.fit_coeff(X_train_batch.float(), encoding_function(y_train_batch, device, num_categories=2))
            # model.fit_coeff(X_train_batch.float(), y_train_batch)

            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)

                # we need y_val_pred_loss for val_loss and y_val_pred for val_acc
                if loss_function == "SmoothL1" or loss_function == "L1" or loss_function == "BCE":
                    y_val_pred_loss = y_val_pred[:, 0].requires_grad_(True).to(device)
                    val_loss = criterion(y_val_pred_loss, y_val_batch.float())
                else:
                    val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch, sigmoid)

                # Predictions and Metrics
                y_val_prob = torch.sigmoid(y_val_pred) if sigmoid else f.softmax(y_val_pred, dim=1)
                y_val_binary = (y_val_prob[:, 1] > 0.5).cpu().numpy()
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
            # print(best_epoch)
            break
        if log_file:
            log_file.write(
                f'Epoch {e:03}: | Train Loss: {train_epoch_loss / len(train_loader):.4f} | Val Loss: {val_epoch_loss / len(val_loader):.4f} | '
                f'Train Acc: {train_epoch_acc / len(train_loader):.4f} | Val Acc: {val_epoch_acc / len(val_loader):.4f} | '
                f'Train Prec: {train_epoch_precision / len(train_loader):.4f} | Val Prec: {val_epoch_precision / len(val_loader):.4f} | '
                f'Train Rec: {train_epoch_recall / len(train_loader):.4f} | Val Rec: {val_epoch_recall / len(val_loader):.4f} | '
                f'Train F1: {train_epoch_f1 / len(train_loader):.4f} | Val F1: {val_epoch_f1 / len(val_loader):.4f} | '
                f'Train AUC: {train_epoch_auc / len(train_loader):.4f} | Val AUC: {val_epoch_auc / len(val_loader):.4f}'
                '\n'
            )
            log_file.flush()
    return best_model, loss_stats['val']
