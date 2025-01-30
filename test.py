import pickle
#import accuracy score, balanced accuracy score, f1 score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
SIMONE_ID=1
FILIPPO_ID=1

def getName():
    return f"Rinaldi Simone: {SIMONE_ID} \n Filippo Brajucha: {FILIPPO_ID}"

def load(clfName):
    if (clfName == "knn"):
        clf = pickle.load(open("knn.save", 'rb'))
        return clf
    elif (clfName == "svr"):
        clf = pickle.load(open("svr.save", 'rb'))
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
    
def predict(df, clfName, clf):
    X = df.drop(["label,type"])
    y = df["type"]
    
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    
    perf= {"accuracy": accuracy, "balanced_accuracy": balanced_accuracy, "f1": f1}
    return perf
    




