<p align="center">
  <img width="400" src="assets/download.webp" >
</p>

# Goal
---

In this notebook, we apply the Intelligent search methods like Differential Evolution Algorithm to find the best ML algorithm hyper-parameters. Previous options are using either predetermined or randomly generated parameters for the ML algorithms. Some of these searching methods are actually a simulation of Intelligent agents in nature like the folk of birds or school of fishes.


### Project Statement
---

### Searching Algorithm of your choice
---

### ML algorithm of your choice
---
Let's use the most Common ML Competition algorithm which is XGBoost.


### Data-set of your choice
---

Let's use the TalkingData set which is available on Kaggle.

### Some handy functions
---

### Preparing a dataset
---
Make balanced data set. Read all 1 values from the train data set and then add the same number of 0 and keep it. Now we do have a balance data set with an equal number of 0 and 1.

```python
df_train = read_train_test_data_balanced(address_train)
df_train.head(3)
```

<div style="text-align:center"><img src="figures/__results___13_0.png" /></div>

### Run DE Algorithm to find the best XGBoost Algorithm hyper-parameters 
---
```python
#Run the DE algorithm on objective function in your favorite range of hyperparameters.
result = list(De_Algorithm(Objective_Function2,
                 [(0.001,1),   #  eta
                  (3,1500),   #  max_leaves
                  (0,20),   #  max_depth
                  (0,1),   #  subsample
                  (0.001,1),   #  colsample_bytree
                  (0.001,1),   #  colsample_bylevel
                  (0.001,1),   #  min_child_weight
                  (2,8),   #  alpha
                  (1,10),   # scale_pos_weight
                  (1,10),     # nthread
                  (1,10)], #  random_state
                  mut=0.4, crossp=0.8, popsize=10, its=40))
```

The best hyper XGBoost Algorithm hyper-parameters found as the follow:
```
eta                    0.355402
max_leaves           520.000000
max_depth             20.000000
subsample              1.000000
colsample_bytree       0.978686
colsample_bylevel      1.000000
min_child_weight       1.000000
alpha                  4.000000
scale_pos_weight       1.000000
nthread                8.000000
random_state           5.000000
Name: 39, dtype: float64
```

### Visualization of searching progress
---

<div style="text-align:center"><img src="figures/__results___23_0.png" /></div>







[Gif reference](https://giphy.com/gifs/c4d-human-ai-8hYQgBIIHkCPjRTmai).
