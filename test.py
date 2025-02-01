#import pickle
import pickle
#import accuracy score, balanced accuracy score, f1 score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
# import pandas
import pandas as pd
#import numpy
import numpy as np

#import label encoder and ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

SIMONE_ID = 1140193 
FILIPPO_ID = 1130613

#return name and code of students
def getName():
    return f"Rinaldi Simone: {SIMONE_ID} \n Filippo Brajucha: {FILIPPO_ID}"

#load the model from the file
def load(clfName):
    if (clfName == "knn"):
        clf = pickle.load(open("knn.save", 'rb'))
        return clf
    elif (clfName == "svm"):
        clf = pickle.load(open("svm.save", 'rb'))
        return clf
    elif (clfName == "rf"):
        clf = pickle.load(open("rf.save", 'rb'))
        return clf
    elif (clfName == "ff"):
        clf = pickle.load(open("ff.save", 'rb'))
        return clf
    elif (clfName == "tb"):
        clf = pickle.load(open("tb.save", 'rb'))
        return clf
    elif (clfName == "tf"):
        clf = pickle.load(open("tf.save", 'rb'))
        return clf
    else:
        return None
    
#preprocess the dataset based on the classifier to use
def preprocess(df, clfName):
    
    #drop column label
    df = df.drop(columns=["label"])
    
    #data cleaning
    df["src_bytes"] = df["src_bytes"].replace("0.0.0.0", np.nan).astype(float)
    mean_src_bytes = df["src_bytes"].mean()
    df["src_bytes"] = df["src_bytes"].fillna(mean_src_bytes)
    
    # casting
    df.astype({'src_bytes': 'int64', 'ts': 'datetime64[ms]', 'dns_AA': 'bool', 'dns_RD': 'bool', 'dns_RA': 'bool', 'dns_rejected': 'bool', 'ssl_resumed': 'bool', 'ssl_established': 'bool', 'weird_notice': 'bool'}).dtypes

    # divide dataset in X and y
    y = df["type"]
    df = df.drop(columns=["type"])

    # Ordinal Encoding
    oe = OrdinalEncoder()
    df_oe = oe.fit_transform(df.select_dtypes(include=['object']))
    df.loc[:, df.select_dtypes(include=['object']).columns] = df_oe
    X = df.to_numpy()

    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # load the scaler
    scaler = pickle.load(open("scaler.save", 'rb'))
    
    # apply the scaler to the dataset
    X = scaler.transform(X)

    # clfs that need PCA
    if ((clfName == "ff") or (clfName == "tb") or (clfName == "tf") or (clfName == "rf")):
        # load the pca
        pca = pickle.load(open("pca.save", 'rb')) 
        # apply the pca to the dataset
        X = pca.transform(X)
        # concatenate X and y
        dfNew = pd.concat([X, y], axis = 1)
        # return the new dataset            
        return dfNew
    
    #clfs that just use Standard Scaling
    elif (clfName == "knn"):
        return pd.concat([X, y], axis = 1)
    
    #clfs that use LDA
    elif (clfName == "svm"):
        # load the lda
        lda = pickle.load(open("lda.save", 'rb'))
        # apply the lda to the dataset
        X = pd.DataFrame(lda.transform(X,y))
        # concatenate X and y
        dfNew = pd.concat([X, y], axis = 1)
        # return the new dataset
        return dfNew
    
    #wrong clf name
    else:
        return None
    
def predict(df, clf):
    # divide dataset in X and y
    y = df["type"]
    X = df.drop(columns=["type"])
    
    # predict the dataset
    y_pred = clf.predict(X)
    
    # compute the accuracy
    accuracy = accuracy_score(y, y_pred)
    
    # compute the balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    
    # compute the f1 score
    f1 = f1_score(y, y_pred, average='weighted')
    
    # create the performance object
    perf = {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "f1": f1}
    
    # return the performance object
    return perf
    


