import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from args import get_args_preprocess

def read_csv_sep(input_path, sep='\t'):
    df_chunk = pd.read_csv(input_path, sep=sep, chunksize=2000)
    res_chunk = []
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res = pd.concat(res_chunk)
    return res

def preprocess_dna_methylation(args):
    root_path = args.data_path
    dna_methylation = read_csv_sep(root_path + 'GDC-PANCAN.methylation450.tsv', sep='\t')
    # dna_methylation = pd.read_csv(root_path + 'GDC-PANCAN.methylation450.tsv', sep='\t', nrows=100)
    # print(dna_methylation.iloc[91])

    samples = pd.read_csv(root_path + 'samples_id.tsv', sep='\t')
    samples = list(samples['sample'])
    samples = ['Composite Element REF'] + samples

    dna_methylation = dna_methylation.filter(items=samples)
    print('org:', dna_methylation)
    print('nansum:', np.sum(dna_methylation.isna().to_numpy()))
    ## 处理nan值 去掉超过10%样本为nan的cg探点

    dna_methylation.dropna(axis=0, thresh=dna_methylation.shape[1] * 0.9, inplace=True)
    # dna_methylation = dna_methylation.fillna(dna_methylation.mean(axis=1), inplace=True)

    ## nan值用当前基因样本平均表达填充
    row_mean = dna_methylation.mean(axis=1).to_numpy()
    values_fill = row_mean.reshape(-1, 1).repeat(dna_methylation.shape[1]-1, 1)
    values_fill = pd.DataFrame(values_fill, index=dna_methylation.index, columns=dna_methylation.columns[1:])
    dna_methylation.fillna(values_fill, inplace=True)

    print('dropnan', dna_methylation)
    print('nansum:', np.sum(dna_methylation.isna().to_numpy()))


    print('loading illuminaMethyl450_hg38')
    illuminaMethyl450_hg38 = read_csv_sep(root_path + 'illuminaMethyl450_hg38_GDC', sep='\t')

    cgs = list(illuminaMethyl450_hg38['#id'])
    cgs_chrom = list(illuminaMethyl450_hg38['chrom'])
    cgs_gene = list(illuminaMethyl450_hg38['gene'])
    # cg_chrom_dict = {cg: chrom for cg, chrom in zip(cgs, cgs_chrom)}
    cg_chrom_dict = {cg: chrom for cg, chrom, gene in zip(cgs, cgs_chrom, cgs_gene) if not gene == '.'}

    chroms = list(illuminaMethyl450_hg38['chrom'].unique())
    chroms_idx = {chrom: [] for chrom in chroms}

    idx_cur = np.array(dna_methylation.index)

    cgs_not_in_hg38 = []
    for i, cg in enumerate(dna_methylation['Composite Element REF']):
        if cg in cg_chrom_dict:
            chroms_idx[cg_chrom_dict[cg]].append(idx_cur[i])
        else:
            cgs_not_in_hg38.append(cg)

    print('not in hg38:', len(cgs_not_in_hg38))

    for chrom, idx_list in chroms_idx.items():
        print(chrom, ':', len(idx_list))

    for chrom, idx_list in chroms_idx.items():
        if not chrom in ['chrX', 'chrY', '*']:
            cur_cgs = np.array(idx_list)
            cur_chr_dna_methylation = dna_methylation.filter(idx_list, axis=0)

            cur_chr_dna_methylation.set_index('Composite Element REF', inplace=True)
            print(chrom, cur_chr_dna_methylation)
            cur_chr_dna_methylation.to_hdfa(args.save_path + 'dna_methylation.h5', key=chrom, complevel=9)


def preprocess_gene_expression(args):
    root_path = args.data_path
    gene_expression_rnaseq = read_csv_sep(root_path + 'GDC-PANCAN.htseq_fpkm-uq.tsv', sep='\t')
    samples = pd.read_csv(root_path + 'samples_id.tsv', sep='\t')
    samples = list(samples['sample'])
    samples = ['xena_sample'] + samples
    gene_expression_rnaseq = gene_expression_rnaseq.filter(items=samples)
    print('org:', gene_expression_rnaseq)

    values = gene_expression_rnaseq.to_numpy()
    idx_cur = gene_expression_rnaseq.index
    idx_drop0 = []
    col_sum_0 = np.sum(values == 0, axis=1)
    for i, s in enumerate(col_sum_0):
        # if s >= gene_expression_rnaseq.shape[1] * 0.95:
        if s >= int(gene_expression_rnaseq.shape[1] * 0.99):
        # if s == gene_expression_rnaseq.shape[1]:
            idx_drop0.append(idx_cur[i])

    gene_expression_rnaseq.drop(index=idx_drop0, inplace=True)
    print('drop 90% 0:', gene_expression_rnaseq)
    print('na_sum', np.sum(gene_expression_rnaseq.isna().to_numpy()))


    gencodev22probeMap = pd.read_csv(root_path + 'gencode.v22.annotation.gene.probeMap', sep='\t')

    genes = list(gencodev22probeMap['id'])
    genes_chrom = list(gencodev22probeMap['chrom'])
    gene_chrom_dict = {gene: chrom for gene, chrom in zip(genes, genes_chrom)}

    idx_cur = np.array(gene_expression_rnaseq.index)
    chrom_idx = {chrom:[] for chrom in list(gencodev22probeMap['chrom'].unique())}
    ensg_not_in_v22 = []
    for i, ensg in enumerate(gene_expression_rnaseq['xena_sample']):
        if ensg in gene_chrom_dict:
            chrom_idx[gene_chrom_dict[ensg]].append(idx_cur[i])
        else:
            ensg_not_in_v22.append(ensg)

    for chrom, idx_list in chrom_idx.items():
        print(chrom, ':', len(idx_list))
    print('ensg_not_in_v22:', len(ensg_not_in_v22))

    idx_target = []
    for chrom, idx_list in chrom_idx.items():
        if not chrom in ['chrY', 'chrX', 'chrM']:
            idx_target += idx_list

    gene_expression_rnaseq = gene_expression_rnaseq.filter(idx_target, axis=0)
    print('drop ensy located at chrY chrX chrM', gene_expression_rnaseq)

    """
    max_min_normailized
    """

    print(gene_expression_rnaseq)
    gene_expression_rnaseq = gene_expression_rnaseq.set_index('xena_sample')
    print(gene_expression_rnaseq)

    gene_expression_rnaseq = (gene_expression_rnaseq - gene_expression_rnaseq.min().min()) / (
                gene_expression_rnaseq.max().max() - gene_expression_rnaseq.min().min())

    gene_expression_rnaseq.to_hdf(args.save_path + 'gene_expression.h5', key='chrs', complevel=9)


def preprocessing_label(args):
    root_path = args.data_path
    basic_phenotype = pd.read_csv(root_path + 'GDC-PANCAN.basic_phenotype.tsv',sep='\t')
    samples_id = pd.read_csv(root_path + 'samples_id.tsv', sep='\t')

    samples = samples_id['sample']
    all_samples = list(basic_phenotype['sample'])

    idx_samples = []

    for sample in samples:
        idx_samples.append(all_samples.index(sample))

    tumor_type = basic_phenotype['project_id']
    sample_type = basic_phenotype['sample_type']

    samples_tumor_type = tumor_type[idx_samples]
    samples_type = sample_type[idx_samples]



    # types = samples_type.unique()
    # print(types)
    # print(len(types))

    # print(len(samples_tumor_type))
    tumor_types = samples_tumor_type.unique()
    # print(tumor_types)
    # print(len(tumor_types))

    # tumor_types_dict = {'tumor_type':list(tumor_types) + ['Normal'],'label':[i for i in range(len(tumor_types))] + [len(tumor_types)]}
    # tumor_types = DataFrame(tumor_types_dict)
    # tumor_types.to_csv(root_path + 'tumor_type_id.tsv',sep='\t',index=False)

    tumor_labels_dict = {t:i for i,t in enumerate(list(tumor_types) + ['Normal'])}

    print(tumor_labels_dict)
    samples_label = []
    samples_tumor_type = list(samples_tumor_type)

    for i, type in enumerate(samples_type):
        if 'Normal' in type:
            label = tumor_labels_dict['Normal']
        else:
            label = tumor_labels_dict[samples_tumor_type[i]]
        samples_label.append(label)

    print(len(samples_label))
    cnts = [0]*34
    for l in samples_label:
        cnts[l] += 1

    # cnts = {tumor:cnt for tumor,cnt in zip(list(tumor_types) + ['Normal'],cnts)}
    # print(cnts)
    # {'TCGA-ACC': 79, 'TCGA-BLCA': 411, 'TCGA-BRCA': 783, 'TCGA-CESC': 306, 'TCGA-CHOL': 36, 'TCGA-COAD': 308,
    #  'TCGA-DLBC': 48, 'TCGA-ESCA': 162, 'TCGA-GBM': 63, 'TCGA-HNSC': 502, 'TCGA-KICH': 65, 'TCGA-KIRC': 321,
    #  'TCGA-KIRP': 274, 'TCGA-LAML': 100, 'TCGA-LGG': 529, 'TCGA-LIHC': 374, 'TCGA-LUAD': 465, 'TCGA-LUSC': 370,
    #  'TCGA-MESO': 86, 'TCGA-OV': 7, 'TCGA-PAAD': 178, 'TCGA-PCPG': 183, 'TCGA-PRAD': 499, 'TCGA-READ': 99,
    #  'TCGA-SARC': 263, 'TCGA-SKCM': 471, 'TCGA-STAD': 338, 'TCGA-TGCT': 156, 'TCGA-THCA': 510, 'TCGA-THYM': 119,
    #  'TCGA-UCEC': 433, 'TCGA-UCS': 56, 'TCGA-UVM': 80, 'Normal': 407}

    # samples_label_dict = {'sample':list(samples),'label':samples_label}
    # samples_label = DataFrame(samples_label_dict)
    # samples_label.to_csv(root_path + 'samples_label.tsv',sep='\t',index=False)


if __name__ == '__main__':
    args = get_args_preprocess()
    if args.dataset in ['exp','both']:
        preprocess_gene_expression(args)
    if args.dataset in ['met', 'both']:
        preprocess_dna_methylation(args)





