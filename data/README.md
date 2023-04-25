# CrossDocked Dataset

1. Download the dataset archive `crossdocked_pocket10.tar.gz` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: `tar -xzvf crossdocked_pocket10.tar.gz`.

# Binding MOAD Dataset

1. Download the dataset with:
```bash
wget http://www.bindingmoad.org/files/biou/every_part_a.zip
wget http://www.bindingmoad.org/files/biou/every_part_b.zip
wget http://www.bindingmoad.org/files/csv/every.csv

unzip every_part_a.zip
unzip every_part_b.zip
```
2. Process the raw data using
python process_bindingmoad.py <bindingmoad_dir>