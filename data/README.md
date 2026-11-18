# Data

Place the four TEP CSV files from [Kaggle](https://www.kaggle.com/datasets/afrniomelo/tep-csv) within this data/ folder. They are ignored by git

| File                       | Runs                      | Samples per run  | Fault starts after sample |
| TEP_FaultFree_Training.csv | 500                       | 500              | none |
| TEP_Faulty_Training.csv    | 500 per fault (20 faults) | 500              | 20 |
| TEP_FaultFree_Testing.csv  | 500                       | 960              | none |
| TEP_Faulty_Testing.csv     | 500 per fault (20 faults) | 960              | 160 |

Columns: `faultNumber`, `simulationRun`, `sample`, then 52 sensors (`xmeas_1` to `xmeas_41`, `xmv_1` to `xmv_11`).

Runs with the same `simulationRun` number in the fault-free and faulty files share a random seed, so they are identical until the fault starts. 
Always split by run number, the same way across all files.
