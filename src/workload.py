from sklearn.base import clone


class Workload:
    '''
    Models each workload. Contains methods to train GPR models on each metric.
    Predicts latency (metric index 0).
    '''
    def __init__(self, wl_id, knobs, metrics, metric_model):
        self.wl_id = wl_id
        self.knobs = knobs
        self.metrics = metrics
        self.metric_model = metric_model
        self.models = {}
        self.N_METRICS = metrics.shape[1]
        
    def build_metric_models(self):
        '''
        Train GPR models on each metric.
        '''
        for metric_idx in range(self.N_METRICS):
            model = clone(self.metric_model)
            model.fit(self.knobs, self.metrics[:, metric_idx])
            self.models[metric_idx] = model
        
    def predict_metric(self, metric_idx, knobs):
        '''
        Predict a metric using existing model.
        '''
        return self.models[metric_idx].predict(knobs)
    