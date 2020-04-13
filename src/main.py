from data_repo import DataRepo
from ottertune import OtterTune
from tester import Tester

from utils import split_train_val

# !!!Only run once, will directly load after
# split_train_val('./data/orig/offline_workload.csv')

dataset = 'split_data' # one of ['sample', 'split_data']
path = '../data/' + dataset + '/'
out_file = '../data/out/' + dataset + '_results.csv' # To write out test results
offline_path = path + 'offline_workload.csv' # offline workload
online_path = path + 'online_workload.csv' # online workload
test_path = path + 'test_knobs.csv' # test knobs (like test.csv)
true_path = path + 'true_latency.csv' # true latency (to measure performance)

repo = DataRepo(offline_path)
o = OtterTune(repo)
t = Tester(o, online_path, test_path, true_path)

t.run(out_file)