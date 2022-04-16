# NGCF

```
pip install -r requirement.txt
```

## File Tree

```
.
├── dataset.py
├── dataset_comp.py
├── make_comp_graph_txt.py
├── make_graph_txt.py
├── metric.py
├── metrics.py
├── model.py
├── models
├── train.py
└── utils.py
```

## Train / Test Split

- `make_comp_graph_txt.py` is making `user items` text file
- `split ratio` is argument and default is 0.1

## Before Training 

- Please `s_adj_matrix.npz`, `train.txt`, `test.txt` move to `/opt/ml/input/data/train`

## Training

```
python train.py --arg1 [arg1] --arg2 [arg2] ...
```

## Inference

- not yet inference