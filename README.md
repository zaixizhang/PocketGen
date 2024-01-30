# :loudspeaker: Fragment wise 3D Structure-based Molecular Generation 

<div align=center>
<img src="./assets/pocketgen.png" width="100%" height="100%" alt="TOC" align=center />
</div>

## Install conda environment via conda yaml file
```bash
conda env create -f pocketgen_env.yaml
conda activate pocketgen_env
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
Pretrained checkpoint: https://drive.google.com/file/d/1nE1YUEPUJ4Szz6VUgxEKROaGQhAAG9_V/view?usp=sharing 
```
python generate_new.py
```





