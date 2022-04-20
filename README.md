# MuchSum

code for "MuchSUM: A Multi-channel Graph Neural Network for Extractive Summarization"

## Introduction

 MuchSUM is a multi-channel graph convolutional network designed to explicitly incorporate multiple salient summary-worthy features. Specifically,
we introduce three specific graph channels to encode the node textual features, node centrality features, and node position features, respectively, under bipartite word-sentence heterogeneous graphs. We also investigate three weighted graphs in each channel to infuse edge features for graph-based summarization modeling.

This project include the code for graph generation. 

![image](Figures/model.png)

## Graph Generation

`gendata.py`: generate graph based on the given data in which sentence is splitted

`discrete_imp.py`:  discretize the importance feature of graph nodes.

`get_importance_index.py`: calculate node importance according to the generated graph

example for input files format:

```json
[
    {"src":[101,1,2,3,101,1,2,3],
    "src_txt":"...",
    "src_sent_labels":[0,1],
    "clss":[0,4]
     // and other contents
    }
]
```

## Dataset

download dataset from https://drive.google.com/drive/folders/1q_piSrtcGZJM1dBV_J15DAD0G2PIQcTn?usp=sharing

include `cnndaily.train.*.pt` and `cnndaily.test.*.pt`

## Citation

```
@article{,
  author    = {Qianren Mao and
               Hongdong Zhu and
               Junnan Li and
               Cheng Ji and
               Hao Peng and
               Jianxin Li and
               Lihong Wang and
               Zheng Wang},
  title     = {MuchSUM: A Multi-channel Graph Neural Network for Extractive Summarization},
  journal   = {},
  volume    = {},
  pages     = {},
  year      = {2022},
  url       = {},
  doi       = {},
  timestamp = {},
  biburl    = {},
  bibsource = {}
```

