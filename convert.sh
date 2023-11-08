#!/bin/bash

# Loop from 0 to 100
for i in $(seq 0 100); do
    input_file="docked_${i}.pdbqt"
    output_file="docked_${i}.sdf"

    # Check if the input file exists before attempting conversion
    if [[ -f "$input_file" ]]; then
        obabel "$input_file" -O "$output_file"
        echo "Converted $input_file to $output_file"
    else
        echo "File $input_file does not exist!"
    fi
done
