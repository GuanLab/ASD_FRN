###################################################################
## Written by Marlena Duda for GuanLab
## USE: generate labeled training matrix
## OUTPUT: brain_matrix_training.txt
## USAGE: python generate_training_matrix.py PATH_TO_BRAIN_MATRIX
###################################################################

import sys

inf = open("asd_truth_set.csv", "rU")
file = inf.readlines()
asd_genes = [line.strip() for line in file]
inf = open("control_gene_set.csv","rU")
file = inf.readlines()
disease_genes = [line.strip() for line in file]

PATH_TO_BRAIN_MATRIX = sys.argv[1]

inf = open(PATH_TO_BRAIN_MATRIX,"r")
file = inf.readlines()
outf = open("brain_matrix_training.txt","w")

for line in file:
    if line == file[0]:
        outf.write("%s\tASD\n"%line.rstrip())
        continue
    chunks = line.split("\t")
    if chunks[0] in asd_genes:
        outf.write("%s\t1\n"%line.rstrip())
    elif chunks[0] in disease_genes:
        outf.write("%s\t0\n"%line.rstrip())

