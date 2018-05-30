## Talk The Walk: Navigating New York City Through Grounded Dialogue
This repository contains code for reproducing experiments
of ```Talk The Walk: Navigating New York City Through Grounded Dialogue``` by Harm de Vries, Kurt Shuster, Dhruv Batra, Devi Parikh, Jason Weston, and Douwe Kiela.

## Getting started

### (1) Setup the environment
```bash
conda env create -f environment.yml
source activate ttw
```

### (2) Download the data
You can download the data by running the following bash script:
```bash
sh scripts/download.sh DATA_DIR
```
where DATA_DIR  (defaults to ./data).

### (3) Run experiments


#### Running emergent language experiments
To reproduce
```bash
python ttw/
```

#### Running natural language experiments


#### Evaluating on full task

#### Running landmark classification experiments


