import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, binarize
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
import time
import os.path
import random
print("Setup Complete")



#necessary functions
def calc_entropy(df, target_col):
    '''
    Given a dataframe and name of target column
    returns entropy value H(x)
    '''
    total_len = len(df)
    target_group_sizes = [] 
    entropy = 0
    for target_val in set(df[target_col]):
        target_group_size = sum(df[target_col]==target_val)
        #print(target_val, target_group_size)
        target_group_sizes.append(target_group_size)
        
        ratio = (target_group_size/total_len)
        entropy -= (ratio * np.log2(ratio))
    #print(entropy)
    return entropy

def info_gain(df, target_col, feat_considered):
    feat_considered_value_set = set(df[feat_considered])
    #print(feat_considered_value_set)
    original_len = len(df)
    parent_entropy = calc_entropy(df, target_col)
    children_entropy = 0
    for val in feat_considered_value_set:
        sub_df = df[df[feat_considered]==val]
        #print(sub_df)
        sub_len = len(sub_df)
        #print(sub_len)
        entropy_i = (sub_len/original_len) * calc_entropy(sub_df, target_col)
        #print(entropy_i)
        children_entropy += entropy_i
    ig = parent_entropy - children_entropy
    return ig

def get_best_binarization_threshold(df, target_col, feat_col):
    best_thresh = None
    best_ig = None
    possible_thresh = set(df[feat_col].values)
    
    #print(dir((possible_thresh)))
    possible_thresh.add(min(possible_thresh)-5)
    possible_thresh.add(max(possible_thresh)+5)
    #print(possible_thresh)
    #print(df[['MonthlyCharges', 'tenure', 'Churn']].head())
    
    for thresh in possible_thresh:
        #print("threshold", thresh)
        df_ = df.copy()
        df_[feat_col] = (df_[feat_col]>thresh).astype(int)
        #print(df_[['MonthlyCharges', 'tenure', 'Churn']].head())
        ig = info_gain(df_, target_col, feat_col)
        #print(ig)
        if (best_thresh==None or ig>best_ig):
            best_thresh = thresh
            best_ig = ig
    return best_thresh

def get_continuous_cols_best_thresh(df, target_col, continuous_cols):
    continuous_cols_best_thresh = {}
    for col in continuous_cols:
        print("getting best threshold for", col, "...")
        best_thresh = get_best_binarization_threshold(df, target_col, col)
        continuous_cols_best_thresh[col] = best_thresh
    return continuous_cols_best_thresh

def plurality_value(values_df):
    #print(type(values_df))
    return values_df.mode()[0]


def make_node(node_type, attribute_chosen, label_value, children):
    node = {}
    node['node_type'] = node_type
    node['attribute_chosen'] = attribute_chosen
    node['label_value'] = label_value
    node['children'] = children
    
    return node

def DecisionTree_Learning(curr_df, parent_df, feature_cols, target_col, original_df, depth, max_depth):
    
    #original_df is taken to extract possible feature values
    
    #create node with 3 properties -> node_type=terminal/internal
    #                       -> attribute_chosen
    #                       -> label_value
    #                       -> children: {} dictionary
    #print(feature_cols, depth)
    #NO examples in curr_df
    if len(curr_df)==0:
        #provide answer from parent
        val = plurality_value(parent_df[target_col])
        return make_node('terminal', None, val, None)
        
    #entropy is 0 in curr_df -> all have same target value
    elif len(set(curr_df[target_col].values)) == 1:
        val = curr_df[target_col].values[0]
        return make_node('terminal', None, val, None)
    
    #no attribute left to select from or max_depth reached
    elif (len(feature_cols)==0 or depth==max_depth):
        val = plurality_value(curr_df[target_col])
        return make_node('terminal', None, val, None)
    
    #splitting required
    else:
        #choose the best attribute to split with 
        ig_max = None
        best_feat = None
        
        for feat in feature_cols:
            ig = info_gain(curr_df, target_col, feat)
            if(ig_max==None or ig>ig_max):
                ig_max = ig
                best_feat = feat
        
        new_feature_cols = feature_cols.copy()
        new_feature_cols.remove(best_feat)
        #print(new_feature_cols)
        possible_vals_of_best_feat = set(original_df[best_feat].values)
        children = {}
        for best_feat_val in possible_vals_of_best_feat:
            child_df = curr_df[curr_df[best_feat]==best_feat_val]
            child_node = DecisionTree_Learning(child_df, curr_df, new_feature_cols, target_col, original_df, depth+1, max_depth)
            children[best_feat_val] = child_node
        val = plurality_value(curr_df[target_col])
        return make_node('internal', best_feat, val, children)
    
def train_DT(df, target_col, max_depth):
    parent_df = None
    feature_cols = list(df.columns)
    feature_cols.remove(target_col)
    original_df = df.copy()
    depth = 0
    root = DecisionTree_Learning(df, parent_df, feature_cols, target_col, original_df, depth, max_depth)
    return root

def predict(X_test, DecisionTreeRoot):
    y_pred = []
    for idx, x_test in X_test.iterrows():
        #print(x_test)
        #predict for each instance
        node = DecisionTreeRoot
        while node['node_type']=='internal':
            #find which attribute is checked and value of that attribute in current test instance
            test_feat_of_node = node['attribute_chosen']
            test_feat_val_of_x_test = x_test[test_feat_of_node]
            #check if the value was faced in training set
            if test_feat_val_of_x_test in node['children']:
                node = node['children'][test_feat_val_of_x_test]
            else:
                #did not face this value while training, so no edge. decide from this node
                break
        y_pred.append(node['label_value'])
    return y_pred

def PreprocessData_telco():
    file_path = "./WA_Fn-UseC_-Telco-Customer-Churn.csv"
    X_original = pd.read_csv(file_path)
    X_original = shuffle(X_original, random_state=2).reset_index(drop=True)
    X_original = X_original.drop(['customerID'], axis=1) #unnecessary attribute

    #make blanks NaN
    X_original.replace(r'^\s*$', np.nan, inplace=True, regex=True)

    #remove rows with missing target
    X_original.dropna(axis=0, how='any', subset=['Churn'], inplace=True)


    #print(X_original.head())
    X = X_original.copy()


    continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


    #X, y - separation, train test split
    y = X.Churn
    X_only = X.drop(['Churn'], axis=1)
    #X_train_, X_test_, y_train, y_test = train_test_split(X_only, y, train_size=0.8, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_only, y, train_size=0.8, test_size=0.2, random_state=0)

    #print(X_only.loc[X_only['MonthlyCharges'] == 46.35])

    #missing value handle 

    # Imputation
#     my_imputer = SimpleImputer(strategy='most_frequent')
#     X_train = pd.DataFrame(my_imputer.fit_transform(X_train_))
#     X_test = pd.DataFrame(my_imputer.transform(X_test_))

    numerical_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    
    pd.options.mode.chained_assignment = None
    
    for col in list(X_train.columns):
        if col in continuous_cols:
            tmp = numerical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = numerical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
        else:
            tmp = categorical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = categorical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
    
    # Imputation removed column names; put them back
#     X_train.columns = X_train_.columns
#     X_test.columns = X_test_.columns


    # encoding categorical values
    # Get list of categorical variables
    label_encoding = True
    if(label_encoding):
        s = (X_train.dtypes == 'object')
        object_cols = list(s[s].index)
        #print(object_cols)

        label_encoder = LabelEncoder()
        for col in object_cols:
            if col not in continuous_cols:
                X_train[col] = label_encoder.fit_transform(X_train[col])
                X_test[col] = label_encoder.transform(X_test[col])
        
        #y_train = pd.Series(label_encoder.fit_transform(y_train), name=y_train.name)
        #y_test = pd.Series(label_encoder.transform(y_test), name=y_test.name)
        mapper = {'Yes': 1, 'No': 0}
        y_train.replace(mapper, inplace=True)
        y_test.replace(mapper, inplace=True)
        
    #reset index and concat
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    Xy_train = pd.concat([X_train, y_train], axis=1)
    
    
    #Datatype changing of continuous columns
    #continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in continuous_cols:
        X_train[col] = pd.to_numeric(X_train[col])
        X_test[col] = pd.to_numeric(X_test[col])
        Xy_train[col] = pd.to_numeric(Xy_train[col])
        
        
    continuous_cols_best_thresh = get_continuous_cols_best_thresh(Xy_train, 'Churn', continuous_cols)
    for feat, thresh in continuous_cols_best_thresh.items():
        X_train[feat] = (X_train[feat]>thresh).astype(int)
        X_test[feat] = (X_test[feat]>thresh).astype(int)
        Xy_train[feat] = (Xy_train[feat]>thresh).astype(int)
    return Xy_train, X_train, X_test, y_train, y_test



def PreprocessData_adult():
    ###########################################TRAIN FILE
    train_file_path = "./adult.data"
    X_original_train = pd.read_csv(train_file_path, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'])
    X_original_train = shuffle(X_original_train, random_state=2).reset_index(drop=True)

    #make blanks NaN
    X_original_train.replace(" \?", np.nan, inplace=True, regex=True)

    #remove rows with missing target
    X_original_train.dropna(axis=0, how='any', subset=['label'], inplace=True)

    #print(X_original.head())
    X_train_with_label = X_original_train.copy()
    
    
    ############################################TEST FILE
    test_file_path = "./adult.test"
    X_original_test = pd.read_csv(test_file_path, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'])
    #X_original_test = shuffle(X_original_test, random_state=2).reset_index(drop=True)

    #make blanks NaN
    X_original_test.replace(" \?", np.nan, inplace=True, regex=True)

    #remove rows with missing target
    X_original_test.dropna(axis=0, how='any', subset=['label'], inplace=True)

    #print(X_original.head())
    X_test_with_label = X_original_test.copy()
    


    
    continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


    

    #############################X, y - separation, train_test_split
    y_train = X_train_with_label.label
    X_train = X_train_with_label.drop(['label'], axis=1)
    y_test = X_test_with_label.label
    X_test = X_test_with_label.drop(['label'], axis=1)
    
    
    ######################## RESOLVE ISSUE WITH Y_TEST LABEL, EXTRA FULLSTOP AT THE END
    #print(set(y_test.values))
    y_test = y_test.str.rstrip('.')
    #print(set(y_test.values))
    
    ##############################missing value handle 

    # Imputation
    numerical_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    
    pd.options.mode.chained_assignment = None
    
    for col in list(X_train.columns):
        if col in continuous_cols:
            tmp = numerical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = numerical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
        else:
            tmp = categorical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = categorical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
    
    # Imputation removed column names; put them back
#     X_train.columns = X_train_.columns
#     X_test.columns = X_test_.columns
    
    # encoding categorical values
    # Get list of categorical variables
    label_encoding = True
    if(label_encoding):
        s = (X_train.dtypes == 'object')
        object_cols = list(s[s].index)
        #print(object_cols)

        label_encoder = LabelEncoder()
        for col in object_cols:
            if col not in continuous_cols:
                X_train[col] = label_encoder.fit_transform(X_train[col])
                X_test[col] = label_encoder.transform(X_test[col])
        
        #y_train = pd.Series(label_encoder.fit_transform(y_train), name=y_train.name)
        #y_test = pd.Series(label_encoder.transform(y_test), name=y_test.name)
        mapper = {' <=50K': 0, ' >50K': 1}
        y_train.replace(mapper, inplace=True)
        y_test.replace(mapper, inplace=True)
    
    
    #reset index and concat
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    Xy_train = pd.concat([X_train, y_train], axis=1)

    
    
    #Datatype changing of continuous columns
    #continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in continuous_cols:
        X_train[col] = pd.to_numeric(X_train[col])
        X_test[col] = pd.to_numeric(X_test[col])
        Xy_train[col] = pd.to_numeric(Xy_train[col])
        
    continuous_cols_best_thresh = get_continuous_cols_best_thresh(Xy_train, 'label', continuous_cols)
    for feat, thresh in continuous_cols_best_thresh.items():
        X_train[feat] = (X_train[feat]>thresh).astype(int)
        X_test[feat] = (X_test[feat]>thresh).astype(int)
        Xy_train[feat] = (Xy_train[feat]>thresh).astype(int)
    return Xy_train, X_train, X_test, y_train, y_test

def PreprocessData_creditcard():
    file_path = "./creditcard.csv"
    X_original = pd.read_csv(file_path)
    #print(X_original.isna().sum())
    
    #remove rows with missing target
    X_original.dropna(axis=0, how='any', subset=['Class'], inplace=True)

    ################################ MAKE X from X_original by taking all positive and 20k negative examples
    X_pos = X_original[X_original['Class']==1]
    X_neg = X_original[X_original['Class']==0]
    #print(len(X_pos), len(X_neg))
    X_neg_20k = X_neg.sample(20000)
    
    X = pd.concat([X_pos, X_neg_20k], axis=0)
    X = shuffle(X, random_state=2).reset_index(drop=True)
    
    X = X.drop(['Time'], axis=1) #unnecessary attribute

    continuous_cols = list(X.columns)
    continuous_cols.remove('Class')
    
    #X, y - separation, train test split
    y = X.Class
    X_only = X.drop(['Class'], axis=1)
    #X_train_, X_test_, y_train, y_test = train_test_split(X_only, y, train_size=0.8, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_only, y, train_size=0.8, test_size=0.2, random_state=0)

    #print(X_only.loc[X_only['MonthlyCharges'] == 46.35])

    #missing value handle 

    # Imputation
#     my_imputer = SimpleImputer(strategy='most_frequent')
#     X_train = pd.DataFrame(my_imputer.fit_transform(X_train_))
#     X_test = pd.DataFrame(my_imputer.transform(X_test_))

    numerical_imputer = SimpleImputer(strategy="mean")
    #categorical_imputer = SimpleImputer(strategy="most_frequent")
    
    pd.options.mode.chained_assignment = None
    
    for col in list(X_train.columns):
        if col in continuous_cols:
            tmp = numerical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = numerical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
        '''
        else:
            tmp = categorical_imputer.fit_transform(X_train[[col]]).ravel()
            X_train[col] = tmp
            tmp = categorical_imputer.transform(X_test[[col]]).ravel()
            X_test[col] = tmp
        '''
    
    # Imputation removed column names; put them back
#     X_train.columns = X_train_.columns
#     X_test.columns = X_test_.columns


    # encoding categorical values
    # Get list of categorical variables
    #################################LABEL ENCODING NOT NEEDED FOR THIS DATASET##############################
    label_encoding = False
    if(label_encoding):
        s = (X_train.dtypes == 'object')
        object_cols = list(s[s].index)
        #print(object_cols)

        label_encoder = LabelEncoder()
        for col in object_cols:
            if col not in continuous_cols:
                X_train[col] = label_encoder.fit_transform(X_train[col])
                X_test[col] = label_encoder.transform(X_test[col])
        
        y_train = pd.Series(label_encoder.fit_transform(y_train), name=y_train.name)
        y_test = pd.Series(label_encoder.transform(y_test), name=y_test.name)
    #########################################################################################################
    
    #reset index and concat
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    Xy_train = pd.concat([X_train, y_train], axis=1)
    
    #Datatype changing of continuous columns
    #continuous_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in continuous_cols:
        X_train[col] = pd.to_numeric(X_train[col])
        X_test[col] = pd.to_numeric(X_test[col])
        Xy_train[col] = pd.to_numeric(Xy_train[col])
        
    continuous_cols_best_thresh = get_continuous_cols_best_thresh(Xy_train, 'Class', continuous_cols)
    for feat, thresh in continuous_cols_best_thresh.items():
        X_train[feat] = (X_train[feat]>thresh).astype(int)
        X_test[feat] = (X_test[feat]>thresh).astype(int)
        Xy_train[feat] = (Xy_train[feat]>thresh).astype(int)
    return Xy_train, X_train, X_test, y_train, y_test

def LoadData_telco():
    if(os.path.exists("./preprocessed/telco_Xy_train.p")):
        print("Preprocessed files found. Loading...")
        Xy_train = pd.read_pickle("./preprocessed/telco_Xy_train.p")
        X_train = pd.read_pickle("./preprocessed/telco_X_train.p")
        y_train = pd.read_pickle("./preprocessed/telco_y_train.p")
        X_test = pd.read_pickle("./preprocessed/telco_X_test.p")
        y_test = pd.read_pickle("./preprocessed/telco_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    else:
        print("Preprocessed files not found. Creating and loading...")
        Xy_train, X_train, X_test, y_train, y_test = PreprocessData_telco()
        Xy_train.to_pickle("./preprocessed/telco_Xy_train.p")
        X_train.to_pickle("./preprocessed/telco_X_train.p")
        y_train.to_pickle("./preprocessed/telco_y_train.p")
        X_test.to_pickle("./preprocessed/telco_X_test.p")
        y_test.to_pickle("./preprocessed/telco_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    
def LoadData_adult():
    if(os.path.exists("./preprocessed/adult_Xy_train.p")):
        print("Preprocessed files found. Loading...")
        Xy_train = pd.read_pickle("./preprocessed/adult_Xy_train.p")
        X_train = pd.read_pickle("./preprocessed/adult_X_train.p")
        y_train = pd.read_pickle("./preprocessed/adult_y_train.p")
        X_test = pd.read_pickle("./preprocessed/adult_X_test.p")
        y_test = pd.read_pickle("./preprocessed/adult_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    else:
        print("Preprocessed files not found. Creating and loading...")
        Xy_train, X_train, X_test, y_train, y_test = PreprocessData_adult()
        Xy_train.to_pickle("./preprocessed/adult_Xy_train.p")
        X_train.to_pickle("./preprocessed/adult_X_train.p")
        y_train.to_pickle("./preprocessed/adult_y_train.p")
        X_test.to_pickle("./preprocessed/adult_X_test.p")
        y_test.to_pickle("./preprocessed/adult_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    
def LoadData_creditcard():
    if(os.path.exists("./preprocessed/creditcard_Xy_train.p")):
        print("Preprocessed files found. Loading...")
        Xy_train = pd.read_pickle("./preprocessed/creditcard_Xy_train.p")
        X_train = pd.read_pickle("./preprocessed/creditcard_X_train.p")
        y_train = pd.read_pickle("./preprocessed/creditcard_y_train.p")
        X_test = pd.read_pickle("./preprocessed/creditcard_X_test.p")
        y_test = pd.read_pickle("./preprocessed/creditcard_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    else:
        print("Preprocessed files not found. Creating and loading...")
        Xy_train, X_train, X_test, y_train, y_test = PreprocessData_creditcard()
        Xy_train.to_pickle("./preprocessed/creditcard_Xy_train.p")
        X_train.to_pickle("./preprocessed/creditcard_X_train.p")
        y_train.to_pickle("./preprocessed/creditcard_y_train.p")
        X_test.to_pickle("./preprocessed/creditcard_X_test.p")
        y_test.to_pickle("./preprocessed/creditcard_y_test.p")
        return Xy_train, X_train, X_test, y_train, y_test
    
def resample(Xy_train, w):
    N = len(Xy_train)
    Xy_train_new = pd.DataFrame(columns=Xy_train.columns)
    #print(Xy_train_new.head())
    w_cumulative = np.full(N, 0.0)
    wsum = 0
    for n in range(N):
        wsum += w[n]
        w_cumulative[n] = wsum
    for n in range(N):
        rand = random.random()
        for iw in range(N):
            if(w_cumulative[iw]>=rand):
                Xy_train_new = Xy_train_new.append(Xy_train.iloc[[iw], :], ignore_index=True)
                break
    return Xy_train_new

def train_adaboost_with_dt(Xy_train, X_train, y_train, target_col, K):
    #Generate same sequence every time
    random.seed(21)
    N = len(Xy_train)
    print(N)
    w = np.full(N, 1.0/N)
    h = []
    z = []
    y_true_on_train_set = list(y_train)
    for k in range(K):
        print("#", k, " stump training...")
        #Xy_train_curr = resample(Xy_train, w)
        Xy_train_copy = Xy_train.copy()
        Xy_train_curr = Xy_train_copy.sample(frac=1, replace=True, weights=w, random_state=0)
        print("sampled size", len(Xy_train_curr))
#         print(len(Xy_train_curr[Xy_train_curr['Class']==0]))
#         print(len(Xy_train_curr[Xy_train_curr['Class']==1]))
        dt_root_curr = train_DT(Xy_train_curr, target_col, 1) # stump
        h.append(dt_root_curr)
        err = 0
        y_pred_on_train_set = predict(X_train, dt_root_curr)
        for j in range(N):
            if y_pred_on_train_set[j]!=y_true_on_train_set[j]:
                err += w[j]
        if(err>0.5):
            print("BAD STUMP")
            z.append(0.0)
        else:
            for j in range(N):
                if y_pred_on_train_set[j]==y_true_on_train_set[j]:
                    w[j] = (w[j]*err)/(1-err)
            w = w / np.sum(w) #NORMALIZATION
            z.append(np.log((1-err)/err))
    return h,z

def adaboost_predict(X_test, h, z):
    len_X_test = len(X_test)
    K = len(z)
    print(len_X_test)
    y_pred_collection = np.zeros(shape=(K,len_X_test))
    
    for k in range(K):
        tmp = predict(X_test, h[k])
        y_pred_collection[k] = np.array(tmp)
        
    y_pred_collection[y_pred_collection==0] = -1 #making 0,1 ---> -1,1
        
    #print(y_pred_collection[:, 130:150])
    z = np.array(z)
    z = z.reshape((1,K))
    y_pred = np.matmul(z, y_pred_collection)
    y_pred = (y_pred>0)*1
    return y_pred.tolist()[0]

def accuracy(tp, tn, fp, fn):
    return (tp+tn)/(tp+tn+fp+fn)
def TP_Rate(tp,tn,fp,fn):
    return tp/(tp+fn)
def TN_Rate(tp,tn,fp,fn):
    return tn/(tn+fp)
def positive_predictive_value(tp,tn,fp,fn):
    return tp/(tp+fp)
def false_discovery_rate(tp,tn,fp,fn):
    return fp/(fp+tp)
def f1_score(tp,tn,fp,fn):
    return (2*tp)/(2*tp+fp+fn)


def performance_measure(y_true, y_pred):
    sz = len(y_true)
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(sz):
        if(y_true[i]==1 and y_pred[i]==1):
            tp+=1
        elif(y_true[i]==0 and y_pred[i]==0):
            tn+=1
        elif(y_true[i]==0 and y_pred[i]==1):
            fp+=1
        elif(y_true[i]==1 and y_pred[i]==0):
            fn+=1
    print("Performance evaluation")
    print("Accuracy:", accuracy(tp,tn,fp,fn))
    print("True positive rate:", TP_Rate(tp,tn,fp,fn))
    print("True negative rate:", TN_Rate(tp,tn,fp,fn))
    print("Positive predictive value:", positive_predictive_value(tp,tn,fp,fn))
    print("False discovery rate:", false_discovery_rate(tp,tn,fp,fn))
    print("F1 score:", f1_score(tp,tn,fp,fn))
    


#main
if __name__ == "__main__":
    try:
        os.makedirs("preprocessed")
    except:
        pass
    
    print(time.ctime(), "Preprocessing...")
    #Xy_train, X_train, X_test, y_train, y_test = LoadData_telco()
    #Xy_train, X_train, X_test, y_train, y_test = LoadData_adult()
    Xy_train, X_train, X_test, y_train, y_test = LoadData_creditcard()

    
    print(time.ctime(), "Preprocessing Done. Training model...")
    dt_flag = False
    if dt_flag:
        #DecisionTreeRoot = train_DT(Xy_train, 'Churn', 6)
        #DecisionTreeRoot = train_DT(Xy_train, 'label', 6)
        #DecisionTreeRoot = train_DT(Xy_train, 'Class', 6)

        print(time.ctime(), "Model trained. Predicting and evaluating...")
        print("############# TRAIN ###################")
        y_true_T = list(y_train)
        y_pred_T = predict(X_train, DecisionTreeRoot)
        performance_measure(y_true_T, y_pred_T)
        print("############# TEST ####################")
        y_true = list(y_test)
        y_pred = predict(X_test, DecisionTreeRoot)
        performance_measure(y_true, y_pred)
        
        #print(y_true[0:20])
        #print(np.unique(y_pred))
        #unique, counts = np.unique(y_true, return_counts=True)
        #print(dict(zip(unique, counts)))
        
        #print(y_pred[0:20])
        #print(np.unique(y_pred))
        #unique, counts = np.unique(y_pred, return_counts=True)
        #print(dict(zip(unique, counts)))
        
    else:
        #h,z = train_adaboost_with_dt(Xy_train, X_train, y_train, 'Churn', 20)
        #h,z = train_adaboost_with_dt(Xy_train, X_train, y_train, 'label', 20)
        h,z = train_adaboost_with_dt(Xy_train, X_train, y_train, 'Class', 5)
        
        print(time.ctime(), "Model trained. Predicting and evaluating...")
        print("############# TRAIN ###################")
        y_true_T = list(y_train)
        y_pred_T = adaboost_predict(X_train, h, z)
        performance_measure(y_true_T, y_pred_T)
        print("############# TEST ####################")
        y_true = list(y_test)
        y_pred = adaboost_predict(X_test, h, z)
        performance_measure(y_true, y_pred)
    
    print(time.ctime(), "Completed.")

