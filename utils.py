import pandas as pd
import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

class Mydataset(Dataset):
    def __init__(self, dna_methylation_list, gene_expression, label):

        self.dna_methylation_list = dna_methylation_list
        self.gene_expression = gene_expression
        self.labels = label
        self.n = len(dna_methylation_list)

    # def __init__(self,root_path_dna,file_path_rna,file_path_label):
        ## 23条染色体上的DNA methylation feature 和  Gene expression feature

        # dna_methylation_list = []
        # for idx in ['chr' + str(i) for i in range(1,22)] + ['chrX']:
        #     file_path_dna = root_path_dna + 'dna_methylation_processed_{}.tsv'.format(idx)
        #     dna_methylation_list.append(pd.read_csv(file_path_dna,sep='\t'))
        #
        # gene_expression = pd.read_csv(file_path_rna,sep='t')
        # labels = pd.read_csv(file_path_label,sep='\t')
        #
        # self.dna_methylation_list = dna_methylation_list
        # self.gene_expression = gene_expression
        # self.labels = labels['label']

    def __len__(self):
        return self.gene_expression.shape[1]

    def __getitem__(self, item):

        labels = self.labels[item]
        labels = torch.LongTensor([labels]).squeeze()

        # print(labels)
        features_list = []
        for i in range(self.n):
            methylation = self.dna_methylation_list[i].iloc[:,item].values
            ## 这里有没有转置的问题 验证一下
            # methylation = torch.FloatTensor(methylation)
            methylation = torch.Tensor(methylation)
            features_list.append(methylation)

        expression = self.gene_expression.iloc[:,item].values
        expression = torch.FloatTensor(expression)
        features_list.append(expression)
        # print(features_list)
        # print(len(features_list))
        # features = torch.stack(features_list,dim=0)
        return features_list, labels

def get_dataloader(args, pretraining=True):
    root_path = args.data_path
    dna_methylation_list = []
    features_dim_list = []

    for idx in ['chr' + str(i) for i in range(1, 23)]:
        print('Loading methylation data on chromosome ' + idx + '...')
        cur_feature = pd.read_hdf(root_path + 'h5/dna_methylation.h5', key=idx)
        dna_methylation_list.append(cur_feature)
        features_dim_list.append(cur_feature.shape[0])


    print('Loading gene expression data...')
    gene_expression = pd.read_hdf(root_path + 'h5/gene_expression.h5', key='chrs')
    features_dim_list.append(gene_expression.shape[0])


    filter_datasets = ['TCGA-ACC','TCGA-BLCA','TCGA-BRCA',
                       # 'TCGA-CESC',
                       'TCGA-CHOL','TCGA-COAD', 'TCGA-DLBC','TCGA-ESCA','TCGA-GBM','TCGA-HNSC','TCGA-KICH','TCGA-KIRC',
                       'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG','TCGA-LIHC','TCGA-LUAD','TCGA-LUSC','TCGA-MESO',
                       # 'TCGA-OV',
                       'TCGA-PAAD', 'TCGA-PCPG',
                       # 'TCGA-PRAD',
                       'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD',
                       # 'TCGA-TGCT',
                       'TCGA-THCA', 'TCGA-THYM',
                       # 'TCGA-UCEC',
                       # 'TCGA-UCS',
                       'TCGA-UVM', 'Normal']

    filter_list = []
    for filter_name in filter_datasets:
        filter = pd.read_csv(root_path + 'sub_dataset/'+filter_name+'_idx.tsv', sep='\t', index_col=0)
        filter = filter['0'].tolist()
        filter_list += filter

    labels = pd.read_csv(root_path + 'samples_label.tsv', sep='\t', index_col=0)

    for i in range(22):
        dna_methylation_list[i] = dna_methylation_list[i].filter(items=filter_list, axis=1)

    gene_expression = gene_expression.filter(items=filter_list, axis=1)

    labels = labels.filter(items=filter_list, axis=0)
    samples_id = np.arange(len(filter_list))

    labels = labels['label'].values

    label_recon = np.zeros([len(labels)])
    for i,d in enumerate(labels):
        c = d
        if d > 3:
            c -= 1
        if d > 19:
            c -= 1
        if d > 22:
            c -= 1
        if d > 27:
            c -= 1
        if d > 30:
            c -= 2
        label_recon[i] = c

    labels = label_recon
    train_index, test_index, train_label, test_label = train_test_split(samples_id, labels,
                                                                        test_size=0.2,
                                                                        # random_state=42,
                                                                        stratify=labels)

    dna_methylation_train = []
    dna_methylation_test = []
    # dna_methylation_val = []
    dna_methylation_full = []

    for i in range(22):
        cur_dna_methylation = dna_methylation_list[i]
        dna_methylation_train.append(cur_dna_methylation.iloc[:, train_index])
        dna_methylation_test.append(cur_dna_methylation.iloc[:, test_index])
        dna_methylation_full.append(cur_dna_methylation)

    gene_expression_train = gene_expression.iloc[:, train_index]
    gene_expression_test = gene_expression.iloc[:, test_index]
    gene_expression_full = gene_expression

    batch_size = args.batch_size
    full_dataset = Mydataset(dna_methylation_full, gene_expression_full, labels)

    if not pretraining:
        train_dataset = Mydataset(dna_methylation_train, gene_expression_train, train_label)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=4)

        test_dataset = Mydataset(dna_methylation_test, gene_expression_test, test_label)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=4)

        full_dataloder = DataLoader(full_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    # shuffle=True,
                                    num_workers=4)
        return train_dataloader, test_dataloader, full_dataloder, features_dim_list

    else:
        train_dataset = Mydataset(dna_methylation_train, gene_expression_train, train_label)
        dataloader_shuf = DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=4)

        dataloader = DataLoader(full_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)

        return dataloader_shuf, dataloader, features_dim_list



def get_sub_dataloader_unsupervised(args, filter_datasets=['TCGA-BRCA', 'Normal']):
    root_path = args.data_path

    dna_methylation_list = []
    features_dim_list = []
    for idx in ['chr' + str(i) for i in range(1, 23)]:
        print('Loading methylation data on chromosome ' + idx + '...')

        cur_feature = pd.read_hdf(root_path + 'h5/dna_methylation.h5', key=idx)
        dna_methylation_list.append(cur_feature)
        features_dim_list.append(cur_feature.shape[0])

    print('Loading gene expression data...')
    gene_expression = pd.read_hdf(root_path + 'h5/gene_expression.h5', key='chrs')
    features_dim_list.append(gene_expression.shape[0])

    filter_list = []
    for filter_name in filter_datasets:
        filter = pd.read_csv(root_path + 'sub_dataset/' + filter_name + '_idx.tsv', sep='\t', index_col=0)
        filter = filter['0'].tolist()
        filter_list += filter

    for i in range(22):
        dna_methylation_list[i] = dna_methylation_list[i].filter(items=filter_list, axis=1)

    gene_expression = gene_expression.filter(items=filter_list, axis=1)

    labels = pd.read_csv(root_path + 'samples_label.tsv', sep='\t', index_col=0)
    labels = labels.filter(items=filter_list, axis=0).to_numpy().squeeze()


    dna_methylation_full = []

    for i in range(22):
        cur_dna_methylation = dna_methylation_list[i]
        dna_methylation_full.append(cur_dna_methylation)

    gene_expression_full = gene_expression

    batch_size = args.batch_size

    full_dataset = Mydataset(dna_methylation_full, gene_expression_full, labels)
    print(len(full_dataset))
    train_dataloader = DataLoader(full_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  # drop_last=True,
                                  num_workers=4)

    full_dataloder = DataLoader(full_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                # shuffle=True,
                                num_workers=4)

    return train_dataloader, full_dataloder, features_dim_list


def load_filtered_samples():
    filter_datasets = ['TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA',
                       # 'TCGA-CESC',
                       'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH',
                       'TCGA-KIRC',
                       'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO',
                       # 'TCGA-OV',
                       'TCGA-PAAD', 'TCGA-PCPG',
                       # 'TCGA-PRAD',
                       'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD',
                       # 'TCGA-TGCT',
                       'TCGA-THCA', 'TCGA-THYM',
                       # 'TCGA-UCEC',
                       # 'TCGA-UCS',
                       'TCGA-UVM', 'Normal']
    filter_list = []
    for filter_name in filter_datasets:
        filter = pd.read_csv('./data/sub_dataset/' + filter_name + '_idx.tsv', sep='\t', index_col=0)
        filter = filter['0'].tolist()
        filter_list += filter
    return filter_list


if __name__ == '__main__':
    get_dataloader()


