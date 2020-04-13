import pandas as pd
from tqdm.autonotebook import tqdm

from sklearn.metrics import mean_squared_error


class Tester:
    '''
    Driver class to run val/test workloads and report performance.
    Each workload required 3 files.
    online_path - Online workloads file
    test_path - Test Knobs file
    true_path - True Latency for test knobs file
    '''
    def __init__(self, ottertune, online_path, test_path, true_path):
        self.ONLINE_PATH = online_path
        self.TEST_PATH = test_path
        self.TRUE_PATH = true_path
        self.o = ottertune
        
        self.online_workloads = {}
        self.test_knobs = {}
        self.true_preds = None
        self.wl_ids = None
        self.__load_data()
        
    def __load_data(self):
        '''
        Run only once at creation of Tester.
        Loads all 3 required files.
        '''
        online = pd.read_csv(self.ONLINE_PATH)
        knobs = pd.read_csv(self.TEST_PATH)
        self.true_preds = pd.read_csv(self.TRUE_PATH, header=None).to_numpy().reshape(-1)
        wl_ids = online['workload id'].unique().tolist()
        self.wl_ids = wl_ids
        for wl_id in tqdm(wl_ids, desc='Loading Online Workloads'):
            w = online[online['workload id'] == wl_id]
            k = knobs[knobs['workload id'] == wl_id].iloc[:, 1:]
            self.online_workloads[wl_id] = w
            self.test_knobs[wl_id] = k
                
    def run(self, out_file):
        '''
        Runs each workload to predict latency for each test knob.
        Saves result file with true/pred workload id and latency.
        Prints MSE across all workloads.
        '''
        preds_arr = []
        pi = 0
        for wl_id in tqdm(self.wl_ids, desc='Running Target Workloads'):
            online_wl = self.online_workloads[wl_id]
            test_knobs = self.test_knobs[wl_id]
            preds, best_wl_id = self.o.predict(online_wl, test_knobs)
            for p in preds: 
                preds_arr.append([wl_id, best_wl_id, self.true_preds[pi], p])
                pi += 1
        
        df = pd.DataFrame(preds_arr, columns=['true_wl_id', 'pred_wl_id', 'true_latency', 'latency_pred'])
        df.to_csv(out_file)
        print('MSE:', mean_squared_error(self.true_preds, df.iloc[:, -1].to_numpy()))
        