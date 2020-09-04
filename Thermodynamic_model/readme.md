# Thermodynamic model

This python script can derive the binding activity of several single- and double- mismatched sgRNAs of CRISPR-dCas9 system based on the thermodynamic properties of base-pairing.

## How to use it
    1. Download Thermo.py and pairs.txt, put them into the same directory;
    2. run "python3 Thermo.py -i <NNNNNNNNNNNNNNNNNNNN> -t pairs.txt -k <temperature(K)>", where "<NNNNNNNNNNNNNNNNNNNN>" refers to the N20 sequence of sgRNA
