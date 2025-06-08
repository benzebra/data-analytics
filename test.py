import pandas as pd
import numpy as np
import itertools, random, time, copy, pickle
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer


SIMONE_ID = 1140193 
FILIPPO_ID = 1130613

PATH = "TRAIN/models"


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')


def test_model(model, criterion, loader):
    model.eval()
    y_pred = torch.tensor([],requires_grad=True).to(device)
    y_true = torch.tensor([],requires_grad=True).to(device)

    total_loss = 0.0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        preds = model(data)
        loss = criterion(preds, targets.long())
        total_loss += loss.item()
        y_pred = torch.cat((y_pred, preds.squeeze()))
        y_true = torch.cat((y_true, targets.detach()))

    avg_loss = total_loss / len(loader)
    return avg_loss, y_pred.squeeze(), y_true.squeeze()

class TabNet(torch.nn.Module):
            def __init__(self, n_d,
                        n_a,
                        n_steps,
                        gamma,
                        optimizer_fn,
                        n_independent,
                        n_shared,
                        epsilon,
                        seed,
                        lambda_sparse,
                        clip_value,
                        momentum,
                        optimizer_params,
                        scheduler_params,
                        mask_type,
                        scheduler_fn,
                        device_name,
                        output_dim,
                        batch_size,
                        num_epochs,
                        unsupervised_model,
                        verbose=0):
                super(TabNet, self).__init__()

                self.batch_size = batch_size
                self.num_epochs = num_epochs
                self.unsupervised_model = unsupervised_model
                self.network = TabNetClassifier(n_d=n_d,
                                                n_a=n_a,
                                                n_steps=n_steps,
                                                gamma=gamma,
                                                optimizer_fn=optimizer_fn,
                                                n_independent=n_independent,
                                                n_shared=n_shared,
                                                epsilon=epsilon,
                                                seed=seed,
                                                lambda_sparse=lambda_sparse,
                                                clip_value=clip_value,
                                                momentum=momentum,
                                                optimizer_params=optimizer_params,
                                                scheduler_params=scheduler_params,
                                                mask_type=mask_type,
                                                scheduler_fn=scheduler_fn,
                                                device_name=device,
                                                output_dim=output_dim,
                                                verbose=verbose)
            
            def fit_model(self, X_train, y_train, X_val, y_val, criterion):
                self.network.fit(X_train=X_train, 
                                y_train=y_train, 
                                eval_set=[(X_train,y_train),(X_val, y_val)], 
                                eval_metric=['accuracy'], 
                                patience=10, 
                                batch_size=self.batch_size, 
                                virtual_batch_size=128, 
                                num_workers=0, 
                                drop_last=True, 
                                max_epochs=self.num_epochs, 
                                loss_fn=criterion, 
                                from_unsupervised=self.unsupervised_model)

            def predict(self, X):
                return self.network.predict(X)
            
            def explain(self, X):
                return self.network.explain(X)
            
            def feature_importances(self):
                return self.network.feature_importances_

            def get_unsupervised_model(n_d_a,n_step,n_independent,n_shared,gamma,lr):
                tabnet_params = dict(n_d=n_d_a, 
                                    n_a=n_d_a,
                                    n_steps=n_step,
                                    gamma=gamma,
                                    n_independent=n_independent,
                                    n_shared=n_shared,
                                    lambda_sparse=1e-3,
                                    optimizer_fn=torch.optim.AdamW, 
                                    optimizer_params=dict(lr=lr),
                                    mask_type="sparsemax",
                                    verbose=0
                                    )
                unsupervised_model = TabNetPretrainer(**tabnet_params)
                return unsupervised_model

class TabTransformer(torch.nn.Module):
            def __init__(self, num_features, num_classes, dim_embedding=8, num_heads=2, num_layers=2):
                super(TabTransformer, self).__init__()
                self.embedding = torch.nn.Linear(num_features, dim_embedding)
                encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
                self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.classifier = torch.nn.Linear(dim_embedding, num_classes)

            def forward(self, x):
                x = self.embedding(x)
                x = x.unsqueeze(1)  # Adding a sequence length dimension
                x = self.transformer(x)
                x = torch.mean(x, dim=1)  # Pooling
                x = self.classifier(x)
                return x

            def predict(self, X):
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Convert input to tensor and move to device
                    outputs = self.forward(X_tensor)
                    _, predictions = torch.max(outputs, 1)  # Get the class with the highest score
                return predictions  # Return predictions as a PyTorch tensor

            def test_model(model, criterion, loader):
                model.eval()
                y_pred = torch.tensor([],requires_grad=True).to(device)
                y_true = torch.tensor([],requires_grad=True).to(device)

                total_loss = 0.0
                
                for data, targets in loader:
                    data, targets = data.to(device), targets.to(device)
                    preds = model(data)
                    loss = criterion(preds, targets.long())
                    total_loss += loss.item()
                    y_pred = torch.cat((y_pred, preds.squeeze()))
                    y_true = torch.cat((y_true, targets.detach()))

                avg_loss = total_loss / len(loader)
                return avg_loss, y_pred.squeeze(), y_true.squeeze()
                    
            def train_model(model, criterion, optimizer, epochs, data_loader, val_loader, device, scheduler, patience):
                n_iter = 0
                best_model = None
                best_val_loss = float('inf')
                epochs_since_last_improvement = 0

                start = time.time()

                loss_history = []
                val_loss_history = []

                for epoch in range(epochs):
                    model.train()

                    start_epoch = time.time()

                    loss_train = 0
                    for data, targets in data_loader:
                        data, targets = data.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(data)
                        targets = targets.long()  # Ensure targets are of type LongTensor
                        if targets.min() < 0 or targets.max() >= outputs.size(1):
                            raise ValueError(f"Target values are out of range. Expected values in [0, {outputs.size(1) - 1}], but got min={targets.min()} and max={targets.max()}.")
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        n_iter += 1
                        loss_train += loss.item()

                    scheduler.step()
                    loss_train /= len(data_loader)

                    # Compute Val Loss
                    val_loss,_,_ = test_model(model, criterion, val_loader)

                    loss_history.append(loss_train)
                    val_loss_history.append(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = copy.deepcopy(model)
                        epochs_since_last_improvement = 0
                    elif epochs_since_last_improvement >= patience:
                        break
                    else:
                        epochs_since_last_improvement += 1

                    print('Epoch [{}/{}] - {:.2f} seconds - train_loss: {:.6f} - val_loss: {:.6f} - patience: {}'.format(epoch+1,
                        epochs, time.time() - start_epoch, loss_train, val_loss, epochs_since_last_improvement), end='\r')

                print('\nTraining ended after {:.2f} seconds - Best val_loss: {:.6f}'.format(time.time() - start, best_val_loss))

                return best_model, loss_history, val_loss_history

class FFNN(torch.nn.Module):
            def __init__(self, input_size, output_size, hidden_size, dropout_prob=0, depth=1):
                super(FFNN, self).__init__()

                self.input = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.Dropout(dropout_prob),
                )

                self.branch1 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(hidden_size),
                    torch.nn.Dropout(dropout_prob),
                )

                for _ in range(depth):
                    self.branch1.extend([
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm1d(hidden_size),
                        torch.nn.Dropout(dropout_prob),
                    ])

                self.branch2 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size//2),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(hidden_size//2),
                    torch.nn.Dropout(dropout_prob),
                )

                for _ in range(depth):
                    self.branch2.extend([
                        torch.nn.Linear(hidden_size//2, hidden_size//2),
                        torch.nn.LeakyReLU(),
                        torch.nn.BatchNorm1d(hidden_size//2),
                        torch.nn.Dropout(dropout_prob),
                    ])
                
                self.output = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size//2, output_size),
                )
                
            def forward(self, x):
                x = self.input(x)
                x = self.branch1(x)
                x = self.branch2(x)
                o  = self.output(x)
                return o
            
            def predict(self, X):
                self.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Convert input to tensor and move to device
                    outputs = self.forward(X_tensor)
                    _, predictions = torch.max(outputs, 1)  # Get the class with the highest score
                return predictions  # Return predictions as a PyTorch tensor


#return name and code of students
def getName():
    return f"Filippo Brajucha: {FILIPPO_ID} \n Simone Rinaldi: {SIMONE_ID}"

#load the model from the file
def load(clfName):
    if (clfName == "knn"):
        clf: KNeighborsClassifier = pickle.load(open(f'{PATH}/knn/knn.pkl', 'rb'))
        return clf
    
    elif (clfName == "svm"):
        clf: SVC = pickle.load(open(f'{PATH}/svm/svm.pkl', 'rb'))
        return clf
    
    elif (clfName == "rf"):
        clf: RandomForestClassifier = pickle.load(open(f'{PATH}/rf/rf.pkl', 'rb'))
        return clf
    
    elif (clfName=="tf"):
        clf: TabTransformer = pickle.load(open(f'{PATH}/tabtransf/tabtransformer.pkl', 'rb'))
        return clf
    
    elif (clfName=="tb"):
        clf: TabNet = pickle.load(open(f'{PATH}/tabnet/tabnet.pkl', 'rb'))
        return clf
    
    elif (clfName=="ff"):
        clf: FFNN = pickle.load(open(f'{PATH}/ffnn/ffnn.pkl', 'rb'))
        return clf
    
    else:
        return None
    
def preprocess(df: pd.DataFrame, clfName: str):
    # 1, 2 e 3
    df = df.dropna()

    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'type' not in df.columns:
        raise KeyError('"type" column is not in the dataframe')

    # 4
    if 'src_bytes' in df.columns:
        df['src_bytes'] = df['src_bytes'].replace('0.0.0.0', np.nan).astype(float)
        mean_src_bytes = df['src_bytes'].mean()
        df['src_bytes'] = df['src_bytes'].fillna(mean_src_bytes)

    # 5
    df = df.astype({'src_bytes': 'int64', 'ts': 'datetime64[ms]', 'dns_AA': 'bool', 'dns_RD': 'bool', 'dns_RA': 'bool', 'dns_rejected': 'bool', 'ssl_resumed': 'bool', 'ssl_established': 'bool'})

    # special characters replacement
    cat_cols = df.select_dtypes(include=['object']).columns
    bool_cols = df.select_dtypes(include=['bool']).columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    mode_cols = (df.select_dtypes(include=['object', 'bool']).columns)
    mode_cols.append(df['ts'].index)
    for col in mode_cols:
        df[col] = df[col].replace('-', df[col].mode()[0])

    for col in num_cols:
        df[col] = df[col].replace('-', df[col].mean())

    X = df.drop(columns=['type'])
    y = df['type']

    # Ordinal Encoding for object and bool columns
    oe_cat: OrdinalEncoder = pickle.load(open(f'{PATH}/preprocessing/ordinal_encoder_cat.pkl', 'rb'))
    oe_bool: OrdinalEncoder = pickle.load(open(f'{PATH}/preprocessing/ordinal_encoder_bool.pkl', 'rb'))
    oe_ts: OrdinalEncoder = pickle.load(open(f'{PATH}/preprocessing/ordinal_encoder_ts.pkl', 'rb'))

    cat_cols = cat_cols.drop('type')
    X[cat_cols] = oe_cat.transform(X[cat_cols])
    X[bool_cols] = oe_bool.transform(X[bool_cols])
    X['ts'] = oe_ts.transform(X['ts'].values.reshape(-1, 1))

    X = pd.get_dummies(X, columns=bool_cols)

    # Label Encoding
    le: LabelEncoder = pickle.load(open(f'{PATH}/preprocessing/label_encoder.pkl', 'rb'))
    y = le.transform(y)
    
    # only std scaling for knn and tb
    std : StandardScaler = pickle.load(open(f'{PATH}/preprocessing/scaler.pkl', 'rb'))
    X = std.transform(X)
            
    # apply PCA for tf, rf, svm and ff
    if clfName == "tf" or clfName == "rf" or clfName == "svm" or clfName == "ff":
        pca: PCA = pickle.load(open(f'{PATH}/preprocessing/pca.pkl', 'rb'))
        X = pca.transform(X)
        
    # Convert X to a DataFrame before concatenating
    X = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])
    y = pd.DataFrame(y, columns=['type'])

    return pd.concat([X, y], axis=1)
        
    

def predict(df, clf):
    print(type(df))
    if isinstance(df, pd.DataFrame):
        X = df.iloc[:, :-1].values  
        y = df.iloc[:, -1].values
    elif isinstance(df, np.ndarray):
        X = df[:, :-1]
        y = df[:, -1]
    else:
        return None

    
    if (isinstance(clf, TabTransformer) or isinstance(clf, FFNN)):
        clf.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_pred = clf.predict(X_tensor).cpu().numpy()
            # y_pred = clf.predict(X_tensor).numpy()
    elif (isinstance(clf, TabNet)):
        y_pred = clf.predict(X)
    else: 
        y_pred = clf.predict(X)
    
    # Calculate metrics
    acc = accuracy_score(y, y_pred)
    bacc = float(balanced_accuracy_score(y, y_pred))
    f1 = f1_score(y, y_pred, average='weighted')

    return {'acc': acc, 'bacc': bacc, 'f1': f1}
    


# if __name__ == '__main__':
#     name = getName()
#     models = ['knn', 'rf', 'svm', 'ff', 'tb', 'tf']
#     data = pd.read_csv('train_dataset.csv', sep=',', low_memory=False)

#     for model in models:
#         dfProcessed = preprocess(data, model)
#         clf = load(model)
#         perf = predict(dfProcessed.values, clf)
#         print(f"{model}: {perf}") 