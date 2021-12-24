# %%
import numpy as np
import math
import pandas as pd
from scipy.stats import entropy
from sklearn import preprocessing, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


np.random.seed(1)


def binarize(x):
    return preprocessing.binarize(x.reshape(-1, 1)).reshape(-1)


def information(y):
    q = len([i for i in y if i == 1])/len(y)
    return entropy([q, 1-q], base=2)


def information_gain(X: np.ndarray, y):
    info = information(y)
    N, n = X.shape
    data = np.column_stack((X, y))
    info_gain = []
    max_bins = 5
    for i in range(n):
        colvals = data[:, i]
        ivals = np.unique(colvals)
        if len(ivals) > max_bins:
            data[:, i] = preprocessing.KBinsDiscretizer(
                n_bins=max_bins, encode='ordinal').fit_transform(colvals.reshape(-1, 1)).reshape(-1)
            ivals = np.unique(colvals)
        infos = []
        weights = []
        for iv in ivals:
            ndata = data[colvals == iv]
            infos.append(information(ndata[:, -1]))
            weights.append(ndata.shape[0]/N)
        info_gain.append(info-np.average(infos, weights=weights))
    return np.array(info_gain)


class LogisticRegression:
    w = np.zeros(0)
    cols = []

    def fit(self, data, n, alpha=0.0001, threshold=0, verbose=False):
        self.w = np.zeros(n+1)
        N = data.shape[0]
        X = data[:, :-1]
        y = data[:, -1]
        y_stretched = y.copy()
        y_stretched[y_stretched == 0] = -1
        selector = SelectKBest(information_gain, k=(
            n if n < X.shape[1] else 'all'))
        selector.fit(X, y)
        self.cols = selector.get_support(indices=True)
        X = X[:, self.cols]
        x = np.column_stack((np.ones(N), X))
        oldloss = N*2
        for i in range(50):
            hw = self.eval(x)
            y_hat = binarize(hw)
            error = metrics.zero_one_loss(y, y_hat)
            loss = np.linalg.norm(y_stretched-hw)
            if verbose:
                print(error, loss)
                # print(metrics.classification_report(y, y_hat))
            if error < threshold:
                break
            self.w += alpha*(((y_stretched-hw)*(1-hw*hw))@x)
            oldloss = loss
        return self

    def eval(self, x):
        return np.tanh(np.atleast_2d(x)@self.w)

    def predict(self, x: np.ndarray):
        x = np.atleast_2d(x)
        x = np.column_stack((np.ones(x.shape[0]), x[:, self.cols]))
        return binarize(self.eval(x))


class AdaBoost:
    learner = None
    h = []
    z = []

    def __init__(self, learner):
        self.learner = learner

    def fit(self, data: np.ndarray, K, n):
        h = self.h = [self.learner() for i in range(K)]
        z = self.z = [0]*K
        N = data.shape[0]
        w = 1/N*np.ones(N)
        x = data[:, :-1]
        y = data[:, -1]
        for k in range(K):
            resampled_data = np.array([data[i]
                                       for i in np.random.choice(range(N), N, p=w)])
            h[k].fit(resampled_data, n, threshold=0.5)
            error = 0
            y_hat = h[k].predict(x)
            misclassifications = y_hat != y
            error = np.dot(w, misclassifications)
            if error > 0.5:
                continue
            w *= (error/(1-error))*(~misclassifications) + misclassifications
            w = preprocessing.normalize([w], norm='l1')[0]
            z[k] = math.log2((1-error)/error)
        return self

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros(N)
        K = len(self.h)
        for i in range(K):
            y_hat = self.h[i].predict(X)
            y_hat[y_hat == 0] = -1
            y += y_hat*self.z[i]
        return binarize(y)


def print_accuracies(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred, zero_division=0))
    print(
        f"False discovery rate:  {1-metrics.precision_score(y_true, y_pred, zero_division=0)}\n\n")


def print_model_accuracies(model, X_train, y_train, X_test, y_test):
    print('training:')
    print_accuracies(y_train, model.predict(X_train))
    print('test:')
    print_accuracies(y_test, model.predict(X_test))


# %%
# dataset = int(input("Enter dataset no: 1. Telco 2. Adult 3. CreditCard 4. Online"))
# K = int(input("Enter number of iterations in adaboost"))

# %%
# https://www.kaggle.com/blastchar/telco-customer-churn
telco = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
# telco
# telco.info()
telco.drop(columns=['customerID'], inplace=True)
# Removing missing values
# telco.dropna(inplace=True)

telco['Churn'].replace(to_replace='Yes', value=1, inplace=True)
telco['Churn'].replace(to_replace='No', value=0, inplace=True)

# numeric columns with missing values can have wrong types if missing values is denoted by a non-empty string
telco.dtypes, telco.columns


# %%
# Fix wrong type of TotalCharges column, will convert incompatible cells to null (np.nan)
telco.TotalCharges = pd.to_numeric(
    telco.TotalCharges, errors='coerce')
telco.isnull().sum()


# %%
y = telco['Churn'].values
X = telco.drop(columns=['Churn'])
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'Partner', 'Dependents',
                        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


# %%
# https://www.kaggle.com/mlg-ulb/creditcardfraud
creditcard = pd.read_csv('./creditcard.csv')
creditcard.drop(columns=['Time'], inplace=True)
creditcard.dtypes


# %%
positives = creditcard[creditcard.iloc[:, -1] == 1]
negatives = creditcard[creditcard.iloc[:, -1] == 0]
creditcard = positives.append(negatives.sample(n=20000))
creditcard.columns


# %%
y = creditcard['Class'].values
X = creditcard.drop(columns=['Class'])
numeric_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                    'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                    'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
categorical_features = []


# %%
# https://www.kaggle.com/sahistapatel96/bankadditionalfullcsv
bankadditionalfull = pd.read_csv('bank-additional-full.csv', sep=';')
bankadditionalfull['y'].replace(to_replace='yes', value=1, inplace=True)
bankadditionalfull['y'].replace(to_replace='no', value=0, inplace=True)
bankadditionalfull.dtypes, bankadditionalfull.columns


# %%
y = bankadditionalfull['y'].values
X = bankadditionalfull.drop(columns=['y'])
numeric_features = ['age', 'duration', 'campaign', 'pdays',
                    'previous', 'emp.var.rate', 'cons.price.idx',
                    'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']


# %%
# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# %%
def adult_read(data):
    data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-per-year']
    # NOTE: dropping a column doesn't change other columns' labels
    data.drop(columns=[data.columns[3]], inplace=True)
    data.replace('?', np.nan, inplace=True)
    # NOTE: experimental knowledge: if using regex to replace, re.sub is used and dtype is not updated
    data.iloc[:, -1].replace(to_replace=['>50K', '>50K.'],
                             value=1, inplace=True)
    data.iloc[:, -1].replace(to_replace=['<=50K', '<=50K.'],
                             value=0, inplace=True)
    print(data.dtypes)
    y = data.iloc[:, -1].values
    X = data.drop(columns=[data.columns[-1]])
    return X, y


# https://archive.ics.uci.edu/ml/datasets/adult
# NOTE: csv can't have space after comma. To use multicharacter separator, use engine='python'.
adult = pd.read_csv('./adult.data', header=None, sep=', ', engine='python')
# adult
# adult.dtypes
adult_test = pd.read_csv(
    './adult.test', skiprows=1, header=None, sep=', ', engine='python')
X_train, y_train = adult_read(adult)
X_test, y_test = adult_read(adult_test)

numeric_features = ['age', 'fnlwgt',
                    'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass',
                        'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# X_train


# %%
numeric_transformer = make_pipeline(SimpleImputer(
    strategy='mean'), preprocessing.RobustScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), preprocessing.OneHotEncoder(
    drop='first', handle_unknown='ignore'))
transformer = make_column_transformer(
    (numeric_transformer, numeric_features), (categorical_transformer, categorical_features), sparse_threshold=0)
transformer.fit(X_train)
X_train = pd.DataFrame(transformer.transform(X_train))
X_test = pd.DataFrame(transformer.transform(X_test))


# %%
model = LogisticRegression().fit(
    np.column_stack((X_train, y_train)), X_train.shape[1])
print('logistic regression:')
print_model_accuracies(model, X_train, y_train, X_test, y_test)


# %%
model = AdaBoost(LogisticRegression).fit(
    np.column_stack((X_train, y_train)), 15, X_train.shape[1])
print('adaboost:')
print_model_accuracies(model, X_train, y_train, X_test, y_test)



