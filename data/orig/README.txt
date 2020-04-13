We provided the following 4 files for you to train the models.

- "offline_workload.CSV" for 58 offline workloads
- "online_workload_B.CSV" for 100 online workloads each with 6 configurations
- "online_workload_C.CSV" for 100 online workloads each with 5 configurations 
- "test.CSV" is the one you need to submit to as as project documentation described.

To note,

1. A configuration includes 12 knobs, denoted as "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "s1", "s2", "s3", "s4". Each knob is a runtime parameter for Apache Spark. For further understanding, please refer to the Spark Documentation.

  "k1": "spark.defalut.parallelism",
  "k2": "spark.executor.instances",
  "k3": "spark.executor.cores",
  "k4": "spark.executor.memory",
  "k5": "spark.reducer.maxsizeInFlight",
  "k6": "spark.shuffle.sort.bypassMergeThreshold",
  "k7": "spark.shuffle.compress",
  "k8": "spark.memory.fraction",
  "s1": "spark.sql.inMemoryColumnarstorage.batchsize",
  "s2": "spark.sql.files.maxPartitionBytes",
  "s3": "spark.sql.autoBroadcastJoinThreshold",
  "s4": "spark.sql.shuffle.partitions"

2. 572 runtime metrics include the latency. The meaning for the name of metrics is beyond the scope of the project.