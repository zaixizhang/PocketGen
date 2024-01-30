# :loudspeaker: PocketGen: Full Atom Protein Pocket Generation with Sequence-Structure Consistency

<div align=center>
<img src="./assets/pocketgen.png" width="100%" height="100%" alt="TOC" align=center />
</div>

## Install conda environment via conda yaml file
```bash
conda env create -f pocketgen_env.yaml
conda activate pocketgen_env
```

## Benchmark Datasets
We use CrossDocked and Binding MOAD datasets to benchmark pocket generation.
```
python data/extract_pocket.py
python split_pl_dataset.py
```

### CrossDocked
We download and process the CrossDocked dataset as described by the authors of [TargetDiff](https://github.com/guanjq/targetdiff)

### Binding MOAD
We download and process the Binding MOAD dataset following the authors of [DiffSBDD](https://github.com/arneschneuing/DiffSBDD)
Download the dataset
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```
Process the raw data using
``` bash
python -W ignore process_bindingmoad.py <bindingmoad_dir>
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





