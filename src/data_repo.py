import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_distances_argmin_min


class DataRepo:
    '''
    Data repository. Contains methods to prune metrics and preprocess knobs.
    '''
    def __init__(self, offline_path, n_factors=2, n_clusters=5, 
                    int_enc=OrdinalEncoder(), cont_enc=MinMaxScaler()):
        self.OFFLINE_WL_PATH = offline_path
        self.METRICS_START_IDX = 14
        self.LATENCY_IDX = 13
        self.INT_KNOBS_IDXS = [9, 10, 11, 12]
        self.CONT_KNOBS_IDXS = [1, 2, 3, 4, 5, 6, 8]
        self.BOOL_KNOS_IDX = 7
        
        self.N_FACTORS = n_factors # no. of factors for Factor Analysis
        self.N_CLUSTERS = n_clusters # k in KMeans
        
        self.pruned_metrics_idxs = None
        self.pruned_metrics_names = None
        self.int_encoder = int_enc # Encoder for integer knobs
        self.cont_encoder = cont_enc # Encoder for continuous knobs
    
    def _build(self):
        '''
        Run only once at the beginning of DataRepo creation.
        Prunes metrics and preprocesses knobs in offline workloads.
        Final processed data is not saved, rather returned to 
        OtterTune to create Workload objects.
        '''
        pruned_data = self.__prune_offline_metrics(self.OFFLINE_WL_PATH)
        processed_data = self.__preprocess_workload_knobs(pruned_data)
        return processed_data
    
    def process_online_workload(self, raw_workload):
        '''
        Prune metrics and preprocess knobs of online workloads.
        '''
        pruned_data = self.__prune_online_metrics(raw_workload)
        return self.__preprocess_workload_knobs(pruned_data, online=True)
    
    def process_test_knobs(self, test_knobs):
        '''
        Preprocess test knobs.
        '''
        return self.__preprocess_workload_knobs(test_knobs, online=True, only_knobs=True)
          
    def __prune_offline_metrics(self, file_path=None):
        '''
        Prune offline workloads metrics using FA + KMeans.
        NOTE: Modularize to use any technique.
        '''
        data = pd.read_csv(file_path)
        metrics = data.to_numpy()[:, self.METRICS_START_IDX:].T

        fa = FactorAnalysis(n_components=self.N_FACTORS)
        metric_factors = fa.fit_transform(metrics)
        km = KMeans(n_clusters=self.N_CLUSTERS).fit(metric_factors)
        closest_idxs, _ = pairwise_distances_argmin_min(km.cluster_centers_, metric_factors)
        self.pruned_metrics_idxs = closest_idxs
        closest_idxs_raw = [self.METRICS_START_IDX + idx for idx in closest_idxs]
        self.pruned_metrics_names = data.columns[closest_idxs_raw].tolist()
        
        pruned_metrics = metrics[self.pruned_metrics_idxs].T
        n_cols = data.shape[1]
        metric_cols = np.linspace(self.METRICS_START_IDX, n_cols - 1, 
                                    n_cols - self.METRICS_START_IDX, dtype=int)
        data.drop(data.columns[metric_cols], axis=1, inplace=True)
        pruned_data = pd.concat([data, pd.DataFrame(pruned_metrics)], axis=1)
        return pruned_data
        
    def __prune_online_metrics(self, raw_workload):
        '''
        Prune online workloads metrics using identified
        non-redundant metrics from offline workloads.
        '''
        data = raw_workload.reset_index(drop=True)
        metrics = data.to_numpy()[:, self.METRICS_START_IDX:].T
        pruned_metrics = metrics[self.pruned_metrics_idxs].T
        
        n_cols = data.shape[1]
        metric_cols = np.linspace(self.METRICS_START_IDX, n_cols - 1, 
                                    n_cols - self.METRICS_START_IDX, dtype=int)
        data.drop(data.columns[metric_cols], axis=1, inplace=True)
        pruned_data = pd.concat([data, pd.DataFrame(pruned_metrics)], axis=1)
        return pruned_data
    
    def __preprocess_workload_knobs(self, pruned_data, online=False, only_knobs=False):
        '''
        Preprocess knobs.
        If online is True, transform using fitted encoders (online knobs)
        Otherwise, fit and then transform (offline knobs)
        For test knobs, only_knobs is True.
        '''
        col_names = pruned_data.columns.tolist()
        pruned_n = pruned_data.to_numpy()
        int_knobs = self.INT_KNOBS_IDXS
        cont_knobs = self.CONT_KNOBS_IDXS
        bool_knob = self.BOOL_KNOS_IDX
        
        if only_knobs:
            int_knobs = [idx - 1 for idx in int_knobs]
            cont_knobs = [idx - 1 for idx in cont_knobs]
            bool_knob = bool_knob - 1
            online = True
        
        
        if not online:
            pruned_n[:, int_knobs] = self.int_encoder.fit_transform(pruned_n[:, int_knobs])
            pruned_n[:, cont_knobs] = self.cont_encoder.fit_transform(pruned_n[:, cont_knobs])
        else:
            pruned_n[:, int_knobs] = self.int_encoder.transform(pruned_n[:, int_knobs])
            pruned_n[:, cont_knobs] = self.cont_encoder.transform(pruned_n[:, cont_knobs])
        
        pruned_n[:, bool_knob] = pruned_n[:, bool_knob].astype(int)
        return pd.DataFrame(pruned_n, columns=col_names)
    