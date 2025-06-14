# 🚀![1727842522286](assets/logo.png)Cuff-KT: Tackling Learners' Real-time Learning Pattern Adjustment via Tuning-Free Knowledge State-Guided Model Updating (KDD2025)

PyTorch implementation of [Cuff-KT](https://arxiv.org/abs/2505.19543).


<h5 align=center>
      
[![arXiv](https://img.shields.io/badge/Arxiv-2505.19543-red?logo=arxiv&label=Arxiv&color=red)](https://arxiv.org/abs/2505.19543)
[![License](https://img.shields.io/badge/Code%20License-MIT%20License-yellow)](https://github.com/zyy-2001/Cuff-KT/blob/master/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/zyy-2001/Cuff-KT)

</h5>

## 🌟Data and Data Preprocessing

Place the [assist15](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data), [assist17](https://sites.google.com/view/assistmentsdatamining/dataset?authuser=0), [comp](https://github.com/wahr0411/PTADisc), [xes3g5m](https://github.com/ai4ed/XES3G5M), and [dbe-kt22](https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH) source files in the dataset directory, and process the data using the following commands respectively:

```python
python preprocess_data.py --data_name assistments15
python preprocess_data.py --data_name assistments17
python preprocess_data.py --data_name comp
python preprocess_data.py --data_name xes3g5m
python preprocess_data.py --data_name dbe_kt22
```

You can also download the dataset from [dataset](https://drive.google.com/drive/folders/1egDh9SZGHrIx1ZHiKZS2udE0-uEFRPa7?usp=sharing) and place it in the `dataset` directory.

The statistics of the five datasets after processing are as follows:

| Datasets | #learners | #questions | #concepts | #interactions |
| :------: | :-------: | :--------: | :-------: | :-----------: |
| assist15 |  17,115  |    100    |    100    |    676,288    |
| assist17 |   1,708   |   3,162   |    411    |    934,638    |
|   comp   |   5,000   |   7,460   |    445    |    668,927    |
| xes3g5m |   5,000   |   7,242   |   1,221   |   1,771,657   |
| dbe-kt22 |   1,186   |    212    |    127    |    306,904    |

## ➡️Quick Start

### Installation

Git clone this repository and create conda environment:

```python
conda create -n cuff python=3.11.9
conda activate cuff
pip install -r requirements.txt 
```

Alternatively, download the environment package from [environment](https://drive.google.com/file/d/1cp88niNVqeITelsCUrOOxJxjPGcxvQQG/view?usp=sharing) and execute the following commands in sequence:

- Navigate to the conda installation directory: /anaconda (or miniconda)/envs/
- Create a folder named `cuff` in that directory
- Extract the downloaded environment package to the conda environment using the command:

```python
tar -xzvf cuff.tar.gz -C /anaconda (or miniconda)/envs/cuff/
conda activate cuff
```

### Training & Testing

You can execute experiments directly using the following commands:

- Controllable Parameter Generation

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp intra --model_name [dkt, atdkt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff --rank 1 --control [ecod, pca, iforest, lof, cuff] --ratio [0, 0.2, 0.4, 0.6, 0.8, 1]
CUDA_VISIBLE_DEVICES=0 python main.py --exp intra --model_name [dkvmn, stablekt, dimkt, diskt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff --rank 1 --control [ecod, pca, iforest, lof, cuff] --ratio [0, 0.2, 0.4, 0.6, 0.8, 1] --convert True
```

- Tuning-Free and Fast Prediction
- - baselines

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22]
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkvmn, stablekt, dimkt, diskt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --convert True
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method [fft, adapter, bitfit]
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkvmn, stablekt, dimkt, diskt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method [fft, adapter, bitfit]  --convert True
```

- - cuff-kt

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff --rank 1
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkvmn, stablekt, dimkt, diskt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff --rank 1 --convert True
```

- Flexible Application

```python
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkt, atdkt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff+ --rank 1
CUDA_VISIBLE_DEVICES=0 python main.py --exp [intra, inter] --model_name [dkvmn, stablekt, dimkt, diskt] --data_name [assistments15, assistments17, comp, xes3g5m, dbe_kt22] --method cuff+ --rank 1 --convert True
```


## ⚠️Citation
If you find our work valuable, we would appreciate your citation: 
```text
@misc{zhou2025cuffkttacklinglearnersrealtime,
      title={Cuff-KT: Tackling Learners' Real-time Learning Pattern Adjustment via Tuning-Free Knowledge State Guided Model Updating}, 
      author={Yiyun Zhou and Zheqi Lv and Shengyu Zhang and Jingyuan Chen},
      year={2025},
      eprint={2505.19543},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.19543}, 
}
```

