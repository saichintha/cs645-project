import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from workload import Workload


class OtterTune:
    '''
    Main OtterTune system. Contains methods to perform workload mapping and predicting latency.
    '''
    def __init__(self, repo, metric_model=GaussianProcessRegressor(kernel=RBF(length_scale=1.0))):
        self.repo = repo
        self.metric_model = metric_model # Used to model each metric in each workload
        self.N_METRICS = None
        
        self.workloads = []        
        self.__build_workloads()
        
        
    def __build_workloads(self):
        '''
        Run only once at OtterTune object creation.
        Creates Workload objects and build metric models on each.
        '''
        print('Building data repository...')
        data = self.repo._build()        
        latency_idx = self.repo.LATENCY_IDX
        wl_ids = data['workload id'].unique()
        
        for wl_id in tqdm(wl_ids, desc='Building Offline Workloads'):
            wl_data = data[data['workload id'] == wl_id].to_numpy()
            knobs = wl_data[:, 1:latency_idx]
            metrics = wl_data[:, latency_idx:]
            if not self.N_METRICS:
                self.N_METRICS = metrics.shape[1]
            workload = Workload(wl_id, knobs, metrics, self.metric_model)
            workload.build_metric_models()
            self.workloads.append(workload)    
    
    def predict(self, raw_workload, test_knobs):
        '''
        Predicts latency for test knobs given online workload.
        Uses helper functions for workload mapping and to
        augment online workload with matched offline workload.
        '''
        processed_wl = self.repo.process_online_workload(raw_workload)
        processed_wl_metrics = processed_wl.iloc[:, 13:]
        processed_wl_knobs = processed_wl.iloc[:, 1:13]
        processed_test_knobs = self.repo.process_test_knobs(test_knobs)

        best_wl_idx = self.__get_best_workload(processed_wl_knobs, processed_wl_metrics)
        aug_wl = self.__get_augmented_workload(best_wl_idx, processed_wl)
        
        gpr = GaussianProcessRegressor(kernel=RBF(length_scale=1.0))
        gpr.fit(aug_wl[:, :-1], aug_wl[:, -1])
        preds = gpr.predict(processed_test_knobs)
        return preds, self.workloads[best_wl_idx].wl_id
    
    def __get_augmented_workload(self, best_wl_idx, processed_wl):
        '''
        Given matched workload, augment current online workload data.
        '''
        w = self.workloads[best_wl_idx]
        w_knobs, w_latency = w.knobs, w.metrics[:, 0].reshape(-1, 1)
        offline = np.concatenate((w_knobs, w_latency), 1)
        
        online = processed_wl.iloc[:, 1:14].to_numpy()
        aug_wl = np.concatenate((offline, online), 0)
        return aug_wl
        
    def __get_best_workload(self, wl_knobs, wl_metrics):
        '''
        Performs workload mapping given online workload (knobs, metrics).
        '''
        n_wls, n_configs = len(self.workloads), len(wl_knobs)
        S = self.__build_distance_matrix(wl_knobs)
        
        binned_S, transf = self.__bin_metrics(S)
        online_metrics = self.__bin_online_metrics(wl_metrics, transf)
        
        best_wl_idx = np.argmin(np.mean(np.sqrt(np.sum((binned_S - online_metrics)**2, axis=2)), axis=0))
        return best_wl_idx
    
    def __build_distance_matrix(self, train_knobs):
        '''
        Build distance matrix S (paper section 6.1).
        Helps efficiently calculate closest offline workload.
        '''
        n_wls, n_configs = len(self.workloads), len(train_knobs)
        S = np.zeros((self.N_METRICS, n_wls, n_configs))
        for metric_idx in range(self.N_METRICS):
            for wl_idx, w in enumerate(self.workloads):
                row = w.predict_metric(metric_idx, train_knobs)
                S[metric_idx, wl_idx, :] = row
        return S

    def __bin_metrics(self, S):
        '''
        Normalizes metrics with bin number using deciles.
        Needed to perform accurate distance comparisons.
        '''
        n_metrics, n_wls, n_configs = S.shape
        sr = S.reshape(n_wls*n_configs, n_metrics)
        transf = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        sr = transf.fit_transform(sr)
        S = sr.reshape(n_metrics, n_wls, n_configs)
        return S, transf
        
    def __bin_online_metrics(self, wl_metrics, transf):
        '''
        Normalizes online metrics with bin number using deciles.
        Uses previsouly used encoder (transf).
        '''
        online_metrics = transf.transform(wl_metrics).T
        online_metrics = np.repeat(online_metrics[:, np.newaxis, :], len(self.workloads), axis=1)
        return online_metrics