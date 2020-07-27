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
















[Gif reference](https://giphy.com/gifs/c4d-human-ai-8hYQgBIIHkCPjRTmai).
