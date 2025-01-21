# 🚀![1727842522286](assets/logo.png)Cuff-KT: Tackling Learners' Real-time Learning Pattern Adjustment via Tuning-Free Knowledge State-Guided Model Updating (ICDE2025 submitted)

PyTorch implementation of [Cuff-KT](https://openreview.net/pdf?id=UVaPEthRKx).

## 🌟Data and Data Preprocessing

Place the [assist15](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data), [comp](https://github.com/wahr0411/PTADisc), and [xes3g5m](https://github.com/ai4ed/XES3G5M) source files in the dataset directory, and process the data using the following commands respectively:

```python
python preprocess_data.py --data_name assistments15
python preprocess_data.py --data_name comp
python preprocess_data.py --data_name xes3g5m
```

The statistics of the three datasets after processing are as follows:

| Datasets | #learners | #questions | #concepts | #interactions |
| :------: | :-------: | :--------: | :-------: | :-----------: |
| assist15 |  17,115  |    100    |    100    |    676,288    |
|   comp   |   5,000   |   7,460   |    445    |    668,927    |
| xes3g5m |   5,000   |   7,242   |   1,221   |   1,771,657   |

## ➡️Quick Start

### Installation

Git clone this repository and create conda environment:

```python
conda create -n cuff-kt python=3.10.13
conda activate cuff-kt
pip install -r requirements.txt 
```

### Training & Testing

Our model experiments are conducted on two NVIDIA RTX 3090 24GB GPUs. You can execute it directly using the following commands:

- Controllable Parameter Generation

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp intra --model_name [dkt, atdkt] --data_name [assistments15, comp, xes3g5m] --method cuff --rank 1 --control [ecod, pca, iforest, lof, cuff] --ratio [0, 0.2, 0.4, 0.6, 0.8, 1] # generator generates parameters for dkt and atdkt
CUDA_VISIBLE_DEVICES=0 python main.py --exp intra --model_name dimkt --data_name [assistments15, comp, xes3g5m] --method cuff --rank 1 --control [ecod, pca, iforest, lof, cuff] --ratio [0, 0.2, 0.4, 0.6, 0.8, 1] --convert True # generator inserts parameters for dimkt
```

- Tuning-Free and Fast Prediction
- - baselines

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt, dimkt] --data_name [assistments15, comp, xes3g5m]
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt, dimkt] --data_name [assistments15, comp, xes3g5m] --method [fft, adapter, bitfit]
```

- - cuff-kt
```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt, dimkt] --data_name [assistments15, comp, xes3g5m] --method cuff --rank 1
```

- Flexible Application

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt, dimkt] --data_name [assistments15, comp, xes3g5m] --method cuff+ --rank 1
```