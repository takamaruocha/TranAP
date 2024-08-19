# **TranAP**

## 1. Overview
We propose Transformer-based Anomaly Prediction (TranAP) method.
TranAP is a forecasting and reconstruction-based multivariate
time series anomaly prediction framework that does not require
anomalous training data. 

## 2. Code Description
- main_TranAP.py: Main file. You can set all parameters.
- evap_TranAP.py: Evaluation file. You can select the trained model from folder `checkpoints/`.
- exp: Training and evaluation folder of TranAP.
- models: Definition folder of TranAP.
- data: Pre-processing folder for datasets.
- datasets: Dataset folder.
- src: Evaluation metrics folder.
- utilities: Other functions folder for training TranAP.
- checkpoints: Save folder for trained models.

## 3. Dataset
We use SWaT, PSM, SMD, SMAP, NIPS-TIS-GECCO datasets.  
You can use `train.csv`, `test.csv`, and `test_label.csv` of the PSM dataset from folder `datasets/PSM/`  
You can download all datasets [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).

## 4. Reproducibility
1. Download data and put them in folder `datasets/`. You can obtain all benchmarks [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR).
2. To train and evaluate TranAP, run:  
```python
python3 main_TranAP.py --data PSM --in_len 48 --out_len 24 --seg_len 12 --itr 5
```
The trained models are saved in folder `checkpoints/`
- Parameter options
```
--data: dataset
--root_path: Root path of the data file
--checkpoints: Location to store the trained model
--in_len: Input length
--out_len: Prediction length
--step_size: Step size
--seg_len: Segment length
--data_dim: Dimensions of data
--d_model: Dimension of hidden states
--d_ff: Dimension of feedforward network
--n_heads: Number of heads
--e_layers: Number of encoder layers
--dropout: Dropout
--attn_ratio: Attention ratio in the attention block
--itr: Experiments times
```
You can see our implementation results in file `log_PSM_il48_ol24.out` 

3. To evaluate the trained model, run:
 ```python
python3 eval_TranAP.py --checkpoint_root ./checkpoints/TranAP/ --setting_name TranAP_PSM_il48_ol24_ss6_sl12_win2_fa10_dm256_nh4_el3_attn0.25_itr0/
```
You can select the trained model from folder `checkpoints/`.
- Parameter options
```
--checkpoint_root: Location of the trained model 
--setting_name: Name of the experiment
```
