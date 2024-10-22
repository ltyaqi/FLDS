# FLAME: Differentially Private Federated Learning in the Shufﬂe Model

This repository contains the code and experiments for the 2021AAAI paper [FLAME: Differentially Private Federated Learning in the Shufﬂe Model](https://arxiv.org/abs/2009.08063)

## Setup

### Dependencies

- install dependencies and activate the virtural environment
```shell
conda env create -f ShuffleFL.yaml
conda activate ShuffleFL
```
- set up the jupyter notebook kernel before plotting figures in it
```shell
python -m ipykernel install --user --name ShuffleFL --display-name ShuffleFL
```

- get the privacy budget for evaluations in Figure 5
```shell
python get_budget.py
```


### Dataset
The [MNIST](https://www.openml.org/d/554) is pre-processed with the basic procedure of standardization. We partition 60,000 samples into 6,000 subsets of 10 samples, with one subset corresponding to a user’s device. 6,000 devices are grouped into 6 batches with size 1,000 (m = 1, 000).
Run the following command to generate train and test data:

```
python generate_data.py
```


The layout the data folder should be:

```
| data
----| openml
---- ----| api
---- ----| data
----| train 
---- ----| train.json
----| test
---- ----| test.json
| generate_data.py
| README.md
| ...
```

## Run
- NP-FL: non-private baseline
```python
python main.py  --optimizer='npsgd'
```

- DP-FL: differentially private baseline (without local privacy)
```python
python main.py  --optimizer='dpsgd'\
                --epsilon=0.237926\
                --delta=5e-6\
                --norm=0.886\
                --mechanism='gaussian'
```

- LDP-FL: locally differentially private baseline
```python
python main.py  --optimizer='ldpsgd'\
                --epsilon=0.237926\
                --delta=5e-6\
                --norm=0.01\
                --mechanism='gaussian'
```

- SS-Simple
```python
python main.py  --optimizer='v1sgd'\
                --epsilon=0.01\
                --norm=0.01\
                --mp_rate=3\
                --mechanism='laplace'
```

- SS-Double
```python
python main.py  --optimizer='v2sgd'\
                --epsilon=0.5\
                --norm=0.01\
                --rate=50\
                --mp_rate=3\
                --mechanism='laplace'
```

- SS-Topk
```python
python main.py  --optimizer='v3sgd'\
                --epsilon=0.5\
                --norm=0.01\
                --rate=50\
                --mp_rate=3\
                --mechanism='laplace'
```

- DP-FL with comparable central DP level when the amplification of subsampling is not counted in FLAME
```python
python main.py  --optimizer='dpsgd'\
                --epsilon=20.5352544\
                --delta=5e-6\
                --norm=0.886\
                --mechanism='guassian'
```

## References
```
@article{liu2020flame,
      title={FLAME: Differentially Private Federated Learning in the Shuffle Model}, 
      author={Ruixuan Liu and Yang Cao and Hong Chen and Ruoyang Guo and Masatoshi Yoshikawa},
      journal={arXiv preprint arXiv:2009.08063},
      year={2020}
}
```
We refer to this [repo](https://github.com/BorjaBalle/amplification-by-shuffling) contributed by the following paper for the numerical evaluation:
```
@inproceedings{balle_privacy_2019,
  title={The privacy blanket of the shuffle model},
  author={Balle, Borja and Bell, James and Gasc{\'o}n, Adri{\`a} and Nissim, Kobbi},
  booktitle={Annual International Cryptology Conference},
  pages={638--667},
  year={2019},
  organization={Springer}
}
```
