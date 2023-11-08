import logomaker
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.figure(figsize=(4, 2))

# Define amino acid sequences
seqs = ['ACVST', 'ACHFT', 'AQEGF', 'ALKFT', 'ACVST', 'RQHGF']

# Create a dictionary to hold the amino acid frequencies
freq_dict = {}

# Loop over each position in the sequences
for i in range(len(seqs[0])):
    # Initialize a dictionary to hold the amino acid counts at this position
    counts = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0}
    # Loop over each sequence and count the amino acids at this position
    for seq in seqs:
        counts[seq[i]] += 1
    # Calculate the frequency of each amino acid at this position
    freqs = {aa: counts[aa] / len(seqs) for aa in counts}
    # Add the frequencies to the frequency dictionary
    freq_dict[i+1] = freqs

# Create a pandas dataframe from the frequency dictionary
data = pd.DataFrame.from_dict(freq_dict, orient='index')

print(data)

# Load the amino acid frequency data
#data = pd.read_csv('amino_acid_frequencies.csv', index_col=0)
data = logomaker.get_example_matrix('ww_information_matrix',
                                     print_description=False)

# Create a Logo object and set the properties
logo = logomaker.Logo(data[17:23], color_scheme='skylign_protein')

# tighten layout
#plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.axis('off')
pp = PdfPages('./logo.pdf')
pp.savefig()
pp.close()

#logo.export('amino_acid_logo.png', format='png')
