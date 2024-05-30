# **TranAP**

## 1. Overview
We propose a forecasting and reconstruction-based multivariate
time series anomaly prediction framework that does not require
anomalous training data. Our Transformer-based Anomaly Prediction (TranAP) 
is trained to predict future trends using only
normal data. 

## 2. Code Description
- main_TranAP.py: The main file. You can set all parameters.
- evap_TranAP.py: The evaluation file. You can select the trained model from folder `checkpoints/`
- exp: The training and evaluation file of TranAP.
- models: The definition folder of TranAP.
- data: The pre-processing folder for datasets.
- datasets: The dataset folder.
- src: The evaluation metrics folder.
- utilities: Other functions for training TranAP.
- checkpoints: The save folder for trained models.

## 3. Dataset
We use SWaT, PSM, SMD, SMAP, NIPS-TIS-GECCO datasets.  
You can use `train.csv`, `test.csv`, `test_label.csv` of the PSM dataset from folder `datasets/PSM/`  
You can download all datasets [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).

## 4. Reproducibility
1. Download data. You can obtain all benchmarks [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).
2. To train and evaluate TranAP, run:  
```python
python3 main_TranAP.py --data PSM --in_len 48 --out_len 24 --seg_len 12 --itr 5
```
The trained models are saved in folder `checkpoints/`
- Parameter options
```
--data: dataset
--root_path: The root path of the data file
--checkpoints: The location to store the trained model
--in_len: The input length
--out_len: The prediction length
--step_size: The step size
--seg_len: The segment length
--data_dim: The dimensions of data
--d_model: The dimension of hidden states
--d_ff: The dimension of feedforward network
--n_heads: The number of heads
--e_layers: The number of encoder layers
--dropout: The dropout
--attn_ratio: The attention ratio in the attention block
--itr: The experiments times
```
3. To evaluate the trained model, run:
 ```python
python3 eval_TranAP.py --checkpoint_root ./checkpoints/TranAP/ --setting_name TranAP_PSM_il48_ol24_ss6_sl12_win2_fa10_dm256_nh4_el3_attn0.25_itr0/
```
You can select the trained model from folder `checkpoints/`
- Parameter options
```
--checkpoint_root: The location of the trained model 
--setting_name: The name of the experiment
```
## 5. Citation
