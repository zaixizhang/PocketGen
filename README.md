## Install conda environment via conda yaml file
```bash
conda env create -f fair_env.yaml
conda activate fair_env
```

## Dataset Processing

```
python data/extract_pocket.py
python split_pl_dataset.py
```


## Training

```
python train_recycle.py
```

## Generation

```
python generate_new.py
```

Pretrained checkpoint: https://drive.google.com/file/d/1nE1YUEPUJ4Szz6VUgxEKROaGQhAAG9_V/view?usp=sharing 

