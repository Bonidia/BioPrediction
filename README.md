![Python](https://img.shields.io/badge/python-v3.7-blue)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-up-brightgreen)

<h1 align="center">
  <img src="https://github.com/Bonidia/BioPrediction/blob/main/misc/img/bioprediction-logo.png" alt="BioPrediction" width="500">
</h1>

<h4 align="center">BioPrediction: Democratizing Machine Learning in the Study of Molecular Interactions</h4>

<h4 align="center">Democratizing Machine Learning in Life Sciences</h4>

<p align="center">
  <a href="https://github.com/Bonidia/BioPrediction">Home</a> •
  <a href="http://autoaipandemics.icmc.usp.br">AutoAI Pandemics</a> •
  <a href="#installing-dependencies-and-package">Installing</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> 
</p>

<h1 align="center"></h1>


## Main Reference:

**Published paper (in Portuguese):** [ENIAC 2023 Anals](https://sol.sbc.org.br/index.php/eniac/article/view/25726/25542)

BioPrediction is part of a bigger project which proposes to democratize Machine Learning in for analysis, study and control of epidemics and pandemics. [Take a look!!!](http://autoaipandemics.icmc.usp.br)

## Abstract

Given the increasing number of biological sequences stored in databases, there is a large source of information that can benefit several sectors such as agriculture and health. Machine Learning (ML) algorithms can extract useful and new information from these data, increasing social and economic benefits, in addition to productivity. However, the categorical and unstructured nature of biological sequences makes this process difficult, requiring ML expertise. In this paper, we propose and experimentally evaluate an end-to-end automated ML-based framework, named BioPrediction, able to identify implicit interactions between sequences, e.g., long non-coding RNA and protein pairs, without the need for end-to-end ML expertise. Our experimental results show that the proposed framework can induce ML models with high predictive accuracy, between 77% and 91%, which are competitive with state-of-the-art tools.

* First study to propose an automated feature engineering and model training pipeline to classify interactions between biological sequences;
    
* The pipeline was mainly tested on datasets regarding lncRNA-protein interactions. The maintainers are further expanding their support to work with other molecules;
    
* BioPrediction can accelerate new studies, reducing the feature engineering time-consuming stage and improving the design and performance of ML pipelines in bioinformatics;
    
* BioPrediction does not require specialist human assistance.

<h1 align="center">
  <img src="https://github.com/Bonidia/BioPrediction/blob/main/misc/img/workflow.png" alt="BioPrediction-Flowchart" width="600">
</h1>


## Maintainers

* Robson Parmezan Bonidia, Bruno Rafael Florentino and Natan Henrique Sanches.

* **Correspondence:** rpbonidia@gmail.com or bonidia@usp.br, brunorf1204@usp.br, natan.sanches@usp.br


## Installing dependencies and package

#### Via miniconda (Terminal)

Installing BioPrediction using Miniconda to manage its dependencies, e.g.:

```sh
$ git clone https://github.com/Bonidia/BioPrediction.git BioPrediction

$ cd BioPrediction

$ git submodule init

$ git submodule update
```

**1 - Install Miniconda:** 

```sh

See documentation: https://docs.conda.io/en/latest/miniconda.html

$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

$ chmod +x Miniconda3-latest-Linux-x86_64.sh

$ ./Miniconda3-latest-Linux-x86_64.sh

$ export PATH=~/miniconda3/bin:$PATH

```

**2 - Create environment:**

```sh

conda env create -f BioPrediction-env.yml -n bioprediction

```

**3 - Activate environment:**

```sh

conda activate bioprediction

```

**4 - You can deactivate the environment using:**

```sh

conda deactivate

```
## How to use

Execute the BioPrediction pipeline with the following command:

```sh
...
To run the code (Example): $ python Bioprediction.py -h

where:

 -input1_fasta_train: txt or fasta format file with the sequences, e.g., data/dataset_1/lncRNA_Sequence.txt
 -input1_fasta_test:  txt or fasta format file with the sequences, e.g., data/dataset_1/lncRNA_Sequence.txt
 -label_1:            the class label for this sequence type, e.g., lncRNA
 -sequence_type1:     type of biological sequence,  e.g., RNA

 -input2_fasta_train: txt or fasta format file with the sequences, e.g., data/dataset_1/protein_Sequence.txt
 -input2_fasta_test:  txt or fasta format file with the sequences, e.g., data/dataset_1/protein_Sequence.txt
 -label_2:            the class label for this sequence type, e.g., enzymes
 -sequence_type2:     type of biological sequence,  e.g., protein

 -interaction_table:  csv format file with the interation matrix, e.g., data/dataset_1/protein_Sequence.txt
 -output: output path, e.g., experiment_1

 -n_cpu:  number of cpus - default = 1
 -estimations: number of estimations - default = 10


execution example:
$ python Bioprediction.py -input1_fasta_train data/dataset_1/lncRNA_Sequence.txt -label_1 LncRNA -sequence_type1 DNA -input2_fasta_train data/dataset_1/protein_Sequence.txt -label_2 rProtein -sequence_type2 Protein -interaction_table data/dataset_1/LPI.csv -output results1


Note Inserting a test dataset is optional.
```

## Citation

If you use this code in a scientific publication, we would appreciate citations to the following paper:

In progress...
