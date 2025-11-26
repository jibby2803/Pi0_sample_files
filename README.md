# Pi0_sample_files
## 🚀 Pi0 torch version

Original code here: https://github.com/huggingface/lerobot.git

This source code use pull [1246](https://github.com/huggingface/lerobot/pull/1246) and change the following files:
1. `lerobot/common/datasets/transforms.py`
2. `lerobot/common/datasets/lerobot_dataset.py`
3. Create folder `calvin` to benchmark with calvin (original source code got some bug? - have not checked again yet)
4. Create folder `libero` to benchmark with libero (take from original source code)
5. Refactor the folder structure and create scripts to easy benchmark with `CALVIN`, `LIBERO`, `SIMPLER_ENV`
6. Add evaluation code for `CALVIN`, `LIBERO`
---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install core dependencies

```bash
conda create -n pitorch python=3.10 -y
conda activate pitorch
pip install -e ".[pi0]"
pip install python-dotenv pytest serial nvitop 
pip install transformers==4.48.1 accelerate datasets==3.0.0

# To run libero simulation for eval
pip install robosuite==1.4.0 bddl gym easydict matplotlib tyro
```
---

## 🧪 Training
```bash
bash script/calvin_finetune.sh
bash script/libero_finetune.sh
bash script/fractal_finetune.sh
```
Several way to write CLI commands, I find this way is the most convenient approach to control the hyperparameters. 
1. The line `cp configs/*.json .....` is really important, all the configs related to `policy` is in here.
2. If you want to change hyperparams of dataset, wandb, stuff... -> checkout `--config_path=...` (the policy config in this file will be overwrited by the line above, so dont worry about it)
3. You also can overwrite config in `configs/*.json` by writting something like "--batch_size=15", checkout `lerobot/configs/policies.py` for more usage.

---

## Evaluation

### 1. Libero evaluation
1. `pip install robosuite==1.4.0 bddl easydict tyro`
2. prepare folder `libero`
```
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
mv LIBERO/libero VLA-Humanoid/
rm -r LIBERO

# must run if you want to extract mask during eval, change to your env path
cp -r tools/binding_utils.py /mnt/lustre-grete/usr/u12045/projects/LLAVA-Med/envs/lerobot/lib/python3.10/site-packages/robosuite/utils/binding_utils.py
cp -r tools/env_wrapper.py libero/libero/envs/env_wrapper.py
cp -r tools/benchmark__init__.py libero/libero/benchmark/__init__.py
```

```
python libero_evaluation.py

# To eval multi gpu
python libero_evaluation_multi_thread.py --exp_name='your_exp'
```
it will log to ./libero_eval_logs, and after finished, you have to see the last score of each file log and compute manually to get the final results

You should get the same results as the following:

| Source                              | Libero 10 | Libero Goal | Libero Object | Libero Spatial | **Average** |
|-------------------------------------|-----------|-------------|---------------|----------------|-------------|
| Our reproduce (chunk 10)            | 85.4      | 93.4        | 96.0          | 90.2           | **91.3**    |
| Report from author SpatialVLA       | 83.0      | 95.0        | 97.0          | 95.0           | **92.5**    |
| Official Paper                      | 85.2      | 95.8        | 98.8          | 96.8           | **94.15**   |


### 2. Calvin evaluation
Checkout the [calvin_readme](./docs/CALVIN_README.md) to setup the environment

Play around with `notebooks/calvin_evaluation.ipynb`
The main evaluation file has not developed yet because I am stuck at the training loss

### 3. Fractal evaluation
Lmao I'm sleepy, see u tomorow!

---

### To do