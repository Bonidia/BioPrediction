#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import argparse
import subprocess
import shutil
import sys
import os.path
import time
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.model_selection import train_test_split 

def make_path(path_input): 
    """Group some strings in a path,
    path_input: a list of all parts of the path"""
    
    path_output = []
    for i in range(len(path_input)):
        path_output.append(os.path.join(path_input[i]))
    
    return path_output

def check_path(paths, type_path='This path'): 
    """Check if this path exist,
    paths: a list of all paths to check its existence
    type_path: path name for identification
    """
    for subpath in paths:
        if os.path.exists(subpath):
            print(f'{type_path} - {subpath}: Found File')
        else:
            print(f'{type_path} - {subpath}: File not exists')
            sys.exit()            

            
class extraction(): #extraction class of the datasets
    
    """Extracts the features from the sequences in the fasta files."""
    
    def __init__(self, ftrain, flabel, ftype, foutput, ftest=''):
        self.ftrain = ftrain
        self.flabel = flabel
        self.ftype = ftype
        self.foutput = foutput
        self.ftest = ftest
        
        self.path = os.path.join(foutput, 'feat_extraction')
        self.path_results = foutput

    def delete_directory(self):
        """Excludes the output directory"""
        try:
            shutil.rmtree(self.path)
            shutil.rmtree(self.path_results)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
            print('Creating Directory...')
    
    def check(self):
        """Checks and creates all output directories"""
    
        if not os.path.exists(self.path_results):
            os.mkdir(self.path_results)

        if not os.path.exists(self.path):
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, 'train'))
            os.mkdir(os.path.join(self.path, 'test'))
    
        if not os.path.exists(os.path.join(self.path, 'train', self.ftype)):
            os.mkdir(os.path.join(self.path, 'train', self.ftype))
            os.mkdir(os.path.join(self.path, 'test', self.ftype))
            
    def extrac_features(self):
        """Uses the input information to extract the features"""
        
        global features_nucleot, features_amino
        
        labels = [self.flabel]
        fasta = [self.ftrain]
        train_size = 0

        if self.ftest:
            labels.append(self.flabel)
            fasta.append(self.ftest)

        datasets = []
        fasta_list = []

        nucl_type = {'dna': '1', 'rna': '2'}
                             
        print(f'{labels[0]} - Extracting features with MathFeature...')
        
        assert self.ftype in ['dna', 'rna', 'protein'], f"Error: Sequence type {self.ftype} not expected "
            
        if self.ftype == "dna" or self.ftype == "rna":

            features_nucleot = [1]#,3,4,5,6,7,8,9,10]
            """Feature extraction for nucleotide-based sequences """    
            for i in range(len(fasta)):
                for j in range(len(fasta[i])):
                    file = fasta[i][j].split('/')[-1] # i: train/test; j: label 1 or 2
                    if i == 0: 
                        preprocessed_fasta = os.path.join(self.path + '/train' + '/'+self.ftype +'/pre_' + file)
                        subprocess.run(['python', 'other-methods/preprocessing.py',
                                        '-i', fasta[i][j], '-o', preprocessed_fasta],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
                    else:  # Test
                        preprocessed_fasta = os.path.join(self.path + '/test'+ '/'+self.ftype +'/pre_' + file)
                        subprocess.run(['python', 'other-methods/preprocessing.py',
                                        '-i', fasta[i][j], '-o', preprocessed_fasta],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    fasta_list.append(preprocessed_fasta)

                    if 1 in features_nucleot:
                        dataset = os.path.join(self.path, 'NAC_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py',
                                        '-i', preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'NAC', '-seq', nucl_type[self.ftype]], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 2 in features_nucleot:
                        dataset = os.path.join(self.path, 'DNC_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'DNC', '-seq', nucl_type[self.ftype]], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                      
                    if 3 in features_nucleot:
                        dataset = os.path.join(self.path, 'TNC_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'TNC', '-seq', nucl_type[self.ftype]], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 4 in features_nucleot:
                        dataset_di = os.path.join(self.path, 'kGap_di_' + self.ftype + '.csv')
                        dataset_tri = os.path.join(self.path, 'kGap_tri_' + self.ftype + '.csv')

                        subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
                                        preprocessed_fasta, '-o', dataset_di, '-l',
                                        labels[i], '-k', '1', '-bef', '1',
                                        '-aft', '2', '-seq', nucl_type[self.ftype]],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                        subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
                                        preprocessed_fasta, '-o', dataset_tri, '-l',
                                        labels[i], '-k', '1', '-bef', '1',
                                        '-aft', '3', '-seq', nucl_type[self.ftype]],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset_di)
                        datasets.append(dataset_tri)

                    if 5 in features_nucleot:
                        dataset = os.path.join(self.path, 'ORF_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/CodingClass.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i]],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 6 in features_nucleot:
                        dataset = os.path.join(self.path, 'Fickett_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/FickettScore.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-seq', nucl_type[self.ftype]], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 7 in features_nucleot:
                        dataset = os.path.join(self.path, 'Shannon_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/EntropyClass.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-e', 'Shannon'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 8 in features_nucleot:
                        dataset =os.path.join(self.path, 'FourierBinary_' + self.ftype + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/FourierClass.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-r', '1'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 9 in features_nucleot:
                        dataset = os.path.join(self.path, 'FourierComplex_' + self.ftype + '.csv')
                        subprocess.run(['python', 'other-methods/FourierClass.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-r', '6'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 10 in features_nucleot:
                        dataset = os.path.join(self.path, 'Tsallis_' + self.ftype + '.csv')
                        subprocess.run(['python', 'other-methods/TsallisEntropy.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

        if self.ftype == "protein": 
            
            features_amino = [1,2]#3,4,5,6,7,8]
            """Feature extraction for aminoacids-based sequences"""    
            for i in range(len(fasta)):
                for j in range(len(fasta[i])):
                    file = fasta[i][j].split('/')[-1]
                    if i == 0:  # Train
                        preprocessed_fasta = os.path.join(self.path + '/train' + '/'+self.ftype +'/pre_' + file)
                        subprocess.run(['python', 'other-methods/preprocessing.py',
                                        '-i', fasta[i][j], '-o', preprocessed_fasta],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        train_size += len([1 for line in open(preprocessed_fasta) if line.startswith(">")])
                    else:  # Test
                        preprocessed_fasta = os.path.join(self.path + '/test'+ '/'+self.ftype +'/pre_' + file)
                        subprocess.run(['python', 'other-methods/preprocessing.py',
                                        '-i', fasta[i][j], '-o', preprocessed_fasta],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    fasta_list.append(preprocessed_fasta)

                    if 1 in features_amino:
                        dataset = os.path.join(self.path, 'Shannon_' + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/EntropyClass.py',
                                        '-i', preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-e', 'Shannon'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 2 in features_amino:
                        dataset = os.path.join(self.path, 'Tsallis_23_'  + self.flabel + '.csv')
                        subprocess.run(['python', 'other-methods/TsallisEntropy.py',
                                        '-i', preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-q', '2.3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 3 in features_amino:
                        dataset = os.path.join(self.path, 'Tsallis_30_'  + self.flabel + '.csv')
                        subprocess.run(['python', 'other-methods/TsallisEntropy.py',
                                        '-i', preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-q', '3.0'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 4 in features_amino:
                        dataset = os.path.join(self.path, 'Tsallis_40_'  + self.flabel + '.csv')
                        subprocess.run(['python', 'other-methods/TsallisEntropy.py',
                                        '-i', preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '5', '-q', '4.0'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 5 in features_amino:
                        dataset = os.path.join(self.path, 'ComplexNetworks_' + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ComplexNetworksClass-v2.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-k', '3'], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 6 in features_amino:
                        dataset_di = os.path.join(self.path, 'kGap_di_' + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/Kgap.py', '-i',
                                        preprocessed_fasta, '-o', dataset_di, '-l',
                                        labels[i], '-k', '1', '-bef', '1',
                                        '-aft', '1', '-seq', '3'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset_di)

                    if 7 in features_amino:
                        dataset = os.path.join(self.path, 'AAC_' + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'AAC'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 8 in features_amino:
                        dataset = os.path.join(self.path, 'DPC_' + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'DPC'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)

                    if 9 in features_amino:
                        dataset = os.path.join(self.path, 'TPC_'  + self.flabel + '.csv')
                        subprocess.run(['python', 'MathFeature/methods/ExtractionTechniques-Protein.py', '-i',
                                        preprocessed_fasta, '-o', dataset, '-l', labels[i],
                                        '-t', 'TPC'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        datasets.append(dataset)
                        
        """Concatenating all the extracted features"""

        assert datasets != [], 'any features was extratc, there is any error' 
        
        datasets = list(dict.fromkeys(datasets))
        dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
        dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
        dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]
        columns = list(dataframes.columns)
        for i in range(1,len(columns)):
            columns[i] = columns[i] + '_' + self.flabel
        dataframes.columns=columns
            
        X_train = dataframes.iloc[:train_size, :]
        X_train.pop('label'+ '_' + self.flabel)
        ftrain = self.path + '/train' + '/'+self.ftype+ '/ftrain_'+self.flabel+'.csv'
        X_train.to_csv(ftrain, index=False)

        ftest = ''
        X_test = []

        if self.ftest:
            X_test = dataframes.iloc[train_size:, :]
            X_test.pop('label'+ '_' + self.flabel)
            ftest = self.path + '/test'+ '/'+self.ftype + '/ftest_'+self.flabel+'.csv'
            X_test.to_csv(ftest, index=False)

        return  X_train, X_test

def make_dataset(data1, data2, label_1, label_2, table, output, arq_name):
    """
    Concatenates both datasets
    
    Args:
    data1 (pandas.DataFrame): First dataset.
    data2 (pandas.DataFrame): Second dataset.
    label_1 (str): Label for the first dataset.
    label_2 (str): Label for the second dataset.
    table (pandas.DataFrame): Interaction table.
    output (str): Output directory.
    arq_name (str): File name for output.

    Returns:
    Tuple of output file paths (str) for data and labels.
    
    """

    data = data1
    for i in range(len(table.columns)-1):

        new_rows= data2.loc[i:i]
        labels = list(new_rows)

        new_rows = new_rows.T
        new_rows = new_rows[i].tolist()
        new_rows = [new_rows[1:len(new_rows)]] * len(table[table.columns])
        data[labels[1:]] = new_rows
        data['Label'] = table[table.columns[i]]
        
        if i == 0:
            dataf = data
        else:
            dataf = pd.concat([dataf, data])
    data_concat = dataf
    data_concat.reset_index()
    data_concat.index = range(len(data_concat.index))

    assert len(data_concat) > 0, 'Interaction table is blank or no sequence pairs were found'

    fnameseq = data_concat.drop('nameseq', axis=1)
    data_concat = data_concat.drop('nameseq', axis=1)
    y = data_concat.pop('Label')

    output_dir = os.path.join(output, 'model_data')
    os.makedirs(output_dir, exist_ok=True)

    foutput_label = os.path.join(output_dir, f'{arq_name}_label.csv')
    foutput_data = os.path.join(output_dir, f'{arq_name}.csv')
    y.to_csv(foutput_label, index=False)
    data_concat.to_csv(foutput_data, index=False)

    return foutput_data, foutput_label, fnameseq


def create_test(foutput_data1, foutput_label1, foutput_data2, foutput_label2, output):
    """
    Creates test dataset by splitting the original dataset.

    Args:
        foutput_data1 (str): File path for the original dataset.
        foutput_label1 (str): File path for the labels of the original dataset.
        foutput_data2 (str): File path for the test dataset.
        foutput_label2 (str): File path for the labels of the test dataset.
        output (str): Output directory.

    Returns:
        Tuple of output file paths (str) for the original and test datasets and their labels.
    """
    foutput_data2 = output + '/model_data' + '/' + 'data_test' + '.csv'
    foutput_label2 = output + '/model_data' + '/' + 'data_test' + '_label.csv'
    X = pd.read_csv(foutput_data1, sep=',')
    y = pd.read_csv(foutput_label1, sep=',')
    foutput_data1, foutput_data2, foutput_label1, foutput_label2 
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=42, shuffle=True)
    
    X_train = X_train.iloc[:len(X_train)//1, :]
    y_train = y_train.iloc[:len(y_train)//1, :]
    X_test =  X_test.iloc[:len(X_test)//1, :] 
    y_test =  y_test.iloc[:len(y_test)//1, :]
    
    X_train.to_csv(foutput_data1, index=False)
    X_test.to_csv(foutput_data2, index=False)
    y_train.to_csv(foutput_label1, index=False)
    y_test.to_csv(foutput_label2, index=False)
    
    return foutput_data1, foutput_label1, foutput_data2, foutput_label2

def objective_rf(space):

    """Automated Feature Engineering - Objective Function - Bayesian Optimization"""
    
    index = list()
    
    descriptors_nucleot = {}
    nucleot_space =[0,4,16,64,320,10,2,5,19,19,5]
    add = 0
    for i in features_nucleot:
        descriptors_nucleot['n'+str(i)] = list(range(add, nucleot_space[i]+add))
        add += nucleot_space[i]
    
    descriptors_amino = {}
    amino_space =[0,5,5,5,5,78,400,20,0,400]
    for i in features_amino:
        descriptors_amino['a'+str(i)] = list(range(add, amino_space[i]+add))
        add += amino_space[i]
    
    dic_type = {'dna': descriptors_nucleot, 'rna': descriptors_nucleot, 'protein': descriptors_amino}
    descriptors = {**dic_type[type1], **dic_type[type2]}
    
 
    for descriptor, ind in descriptors.items():
        if int(space[descriptor]) == 1:
            index = index + ind

    x = df_x.iloc[:, index]

    if int(space['Classifier']) == 0:
        if len(fasta_label_train) > 2:
            model = AdaBoostClassifier(random_state=63) #MUDAR
        else:
            model = CatBoostClassifier(n_estimators=500,
                                       thread_count=n_cpu, nan_mode='Max',
                                       logging_level='Silent', random_state=63)
    elif int(space['Classifier']) == 1:
        model = RandomForestClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)
    else:
        model = lgb.LGBMClassifier(n_estimators=500, n_jobs=n_cpu, random_state=63)

    if len(fasta_label_train) > 2:
        score = make_scorer(f1_score, average='weighted')
    else:
        score = make_scorer(balanced_accuracy_score)

    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    metric = cross_val_score(model,
                             x,
                             labels_y,
                             cv=kfold,
                             scoring=score,
                             n_jobs=n_cpu).mean()

    return {'loss': -metric, 'status': STATUS_OK}      

        
def feature_engineering(estimations, train, train_labels, test, foutput):

    """Automated Feature Engineering - Bayesian Optimization"""

    global df_x, labels_y

    print('Automated Feature Engineering - Bayesian Optimization')

    df_x = pd.read_csv(train)
    labels_y = pd.read_csv(train_labels)

    if test != '':
        df_test = pd.read_csv(test)

    path_bio = foutput + '/best_descriptors'
    if not os.path.exists(path_bio):
        os.mkdir(path_bio)
    
    param_nucleot = {}
    for i in features_nucleot:
        param_nucleot['n'+str(i)] = [0, 1]
    
    param_amino = {}
    for i in features_amino:
        param_amino['a'+str(i)] = [0, 1]
    
    param_classif = {'Classifier': [0, 1, 2]}
    
    param_type = {'dna': param_nucleot, 'rna': param_nucleot, 'protein': param_amino}
    param = {**param_type[type1], **param_type[type2], **param_classif}
    
    space_nucleot = {}
    for i in features_nucleot:
        space_nucleot['n'+str(i)] = hp.choice('n'+str(i), [0, 1])
    
    space_amino = {}
    for i in features_amino:
        space_amino['a'+str(i)] = hp.choice('a'+str(i), [0, 1])

    space_classif = {'Classifier': hp.choice('Classifier', [0, 1, 2])}
    space_type = {'dna': space_nucleot, 'rna': space_nucleot, 'protein': space_amino}
    space = {**space_type[type1], **space_type[type2], **space_classif}
    
    trials = Trials()
    best_tuning = fmin(fn=objective_rf,
                space=space,
                algo=tpe.suggest,
                max_evals=estimations,
                trials=trials)

    index = list()
    descriptors_nucleot = {}
    nucleot_space =[0,4,16,64,320,19,2,5,19,19,5]
    add = 0
    for i in features_nucleot:
        descriptors_nucleot['n'+str(i)] = list(range(add, nucleot_space[i]+add))
        add += nucleot_space[i]

    descriptors_amino = {}
    amino_space =[0,5,5,5,5,78,400,20,0,400]
    for i in features_amino:
        descriptors_amino['a'+str(i)] = list(range(add, amino_space[i]+add))
        add += amino_space[i]

    dic_type = {'dna': descriptors_nucleot, 'rna': descriptors_nucleot, 'protein': descriptors_amino}
    descriptors = {**dic_type[type1], **dic_type[type2]}

    for descriptor, ind in descriptors.items():
        result = param[descriptor][best_tuning[descriptor]]
        if result == 1:
            index = index + ind

    classifier = param['Classifier'][best_tuning['Classifier']]

    btrain = df_x.iloc[:, index]
    path_btrain = path_bio + '/best_train.csv'
    btrain.to_csv(path_btrain, index=False, header=True)

    if test != '':
        btest = df_test.iloc[:, index]
        path_btest = path_bio + '/best_test.csv'
        btest.to_csv(path_btest, index=False, header=True)
    else:
        btest, path_btest = '', ''

    return classifier, path_btrain, path_btest, btrain, btest 

#################################################################################################################################
#################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-input1_fasta_train', '--input1_fasta_train',  nargs='+', help='fasta format file, e.g., fasta/ncRNA.fasta')
parser.add_argument('-input1_fasta_test', '--input1_fasta_test',  nargs='+', help='fasta format file, e.g., fasta/ncRNA.fasta')
parser.add_argument('-label_1', '--label_1', help='problem labels, e.g., LncRNA')
parser.add_argument('-sequence_type1', '--sequence_type1', help='sequence type problem, e.g., RNA')

parser.add_argument('-input2_fasta_train', '--input2_fasta_train', nargs='+', help='fasta format file, e.g., fasta/protein.fasta')
parser.add_argument('-input2_fasta_test', '--input2_fasta_test', nargs='+', help='fasta format file, e.g., fasta/protein.fasta')
parser.add_argument('-label_2', '--label_2', help='problem labels, e.g., Regulatory_proteins')
parser.add_argument('-sequence_type2', '--sequence_type2', help='sequence type problem, e.g., Protein')

parser.add_argument('-interaction_table', '--interaction_table', help='txt format file, e.g., interactions/table.txt')
parser.add_argument('-output', '--output', help='resutls directory, e.g., result/')

parser.add_argument('-n_cpu', '--n_cpu', default=1, help='number of cpus - default = 1')
parser.add_argument('-estimations', '--estimations', default=20, help='number of estimations - BioAutoML - default = 10')

################################################## inputs
args = parser.parse_args()
input1_fasta_train = args.input1_fasta_train
input1_fasta_test = args.input1_fasta_test
label_1 = args.label_1
type1 = args.sequence_type1.lower()

input2_fasta_train = args.input2_fasta_train
input2_fasta_test = args.input2_fasta_test
label_2 = args.label_2
type2 = args.sequence_type2.lower()

interaction_table = args.interaction_table
foutput = args.output

estimations = int(args.estimations)
n_cpu = int(args.n_cpu)

fasta_label_train = [['0'],['1']]
 
################################################# check and make paths
check_path(input1_fasta_train,'input1_fasta_train') 
check_path(input2_fasta_train,'input2_fasta_train')
input1_fasta_train = make_path(input1_fasta_train)
input2_fasta_train = make_path(input2_fasta_train)

if None != input1_fasta_test and None != input2_fasta_test:
    check_path(input1_fasta_test,'input1_fasta_test') 
    check_path(input2_fasta_test,'input2_fasta_test')
    input1_fasta_test = make_path(input1_fasta_test)
    input2_fasta_test = make_path(input2_fasta_test) 
else:
    print('Any test input')
    
check_path([interaction_table],'interaction_table')
table = pd.read_csv(interaction_table, sep=',')
                                            
################################################# general extraction
input1 = extraction(input1_fasta_train, label_1, type1, foutput, ftest=input1_fasta_test)
input1.delete_directory()
input1.check()        
train1, test1 = input1.extrac_features()

input2 = extraction(input2_fasta_train, label_2, type2, foutput, ftest=input2_fasta_test)   
input2.check()        
train2, test2 = input2.extrac_features()

################################################# concatenation
foutput_data1, foutput_label1, fnameseqtrain = make_dataset(train1,train2,label_1,label_2, table, foutput, 'data_train')
if input1_fasta_test:
    foutput_data2, foutput_label2, fnameseqtest = make_dataset(test1,test2,label_1,label_2, table, foutput, 'data_test')
else:
    foutput_data2, foutput_label2, fnameseqtest = '','',''
    foutput_data1, foutput_label1, foutput_data2, foutput_label2 = create_test(foutput_data1, foutput_label1, foutput_data2, foutput_label2, foutput)

################################################# feature engineering and binary bioautoml 
classifier, path_train, path_test, train_best, test_best = \
        feature_engineering(estimations, foutput_data1, foutput_label1, foutput_data2, foutput)

classifier = 2
subprocess.run(['python', 'BioAutoML-binary.py', '-train', path_train,
                     '-train_label', foutput_label1, '-test', path_test, '-test_label',
                     foutput_label2, '-test_nameseq', fnameseqtest, '-imbalance', 'True',
                     '-nf', 'True', '-classifier', str(classifier), '-n_cpu', str(n_cpu),
                     '-output', foutput])

