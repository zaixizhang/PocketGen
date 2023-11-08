This is a preliminary version of Pocket Generation for reference. 
## Install conda environment via conda yaml file
```bash
conda env create -f fair_env.yaml
conda activate fair_env
```

## Datasets
Please refer to the `data` folder.


## Training

```
python train_recycle.py
```

## Generation

```
python data/optimization_mp.py
```

