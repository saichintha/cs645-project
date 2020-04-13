import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_val(offline_path, save_path='./data/split_data/'):
    '''
    Splits original offline workload into train and val sets
    Random configs of each workload are split into train (offline) and val (online)
    Random configs of val are further split into val and test (just the knobs).
    NOTE: Messy, will clean up later
    '''
    offline = pd.read_csv(offline_path)
    X, y = offline, offline['latency']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=645)
    X_train.sort_values('workload id').to_csv(save_path + 'offline_workload.csv', index=False)
    XT_train, XT_test, yT_train, yT_test = train_test_split(X_test, y_test, test_size=0.30, random_state=645)
    XT_train.sort_values('workload id').to_csv(save_path + 'online_workload.csv', index=False)
    XT_test.iloc[:, :13].sort_values('workload id').to_csv(save_path + 'test_knobs.csv', index=False)
    XT_test.iloc[:, [0,13]].sort_values('workload id').iloc[:, 1].to_csv(save_path + 'true_latency.csv', index=False, header=False)
    
    # Of the 58 workloads, how many are represented in test workloads
    # Making sure we test at least one config in each workload
    print(XT_train['workload id'].nunique(), XT_test['workload id'].nunique())
    
# Only run once, will directly load after
# split_train_val('./data/orig/offline_workload.csv')