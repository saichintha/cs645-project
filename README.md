# Workflow

#### DataRepo
* Load offline workloads, prune metrics, pre-process knobs.
* Saves pruned metrics, knob encoders to be used later to transform online workloads.

#### OtterTune 
* Create offline Workload objects from processed offline data from DataRepo  
* Given online workload and test knobs, performs workload mapping
* Augments current workload with mapped workload
* Predict latency for test knobs  
* Uses pruned metric info and trained knob encoders in DataRepo to transform online workloads

#### Workload
* Models a single workload
* Contains GPR models trained on each metric (knobs -> GPR -> latency/metric)

#### Tester
* Works given a mode ('val' or 'test')
* Loads online workloads and their respective test knobs
* If 'val', loads true latencies to report mean absolute error (MAE)
* Saves a result file under `./data/out/` with mapped workload info and predicted latency
* Result file name `val_results_({MAE}).csv`

#### Datasets
* Train - offline_workload.csv (makes our DataRepo)
* Val – online_workload_B.csv (100 workloads with 6 configs each) split into 3 files
    1. `online_workload.csv` (100 workloads with 5 configs each, randomly chosen)
    2. `test_knobs.csv` (100 workloads with left out 1 config each)
    3. `true_latency.csv` (True latency values for each test knob in test_knobs.csv)
* Test – online_workload_C.csv and provided test knobs
