# AVBAE-MODFR
The code and dataset for AVBAE-MODFR
# 1. Environment setup
We recommend you build a Python virtual environment with Anaconda.
## 1.1 Create and activate a new virtual environment
```
conda create -n AVBAE-MODFR python=3.7
conda activate AVBAE-MODFR
```
## 1.2 Install the package
```
pip install -r requirements.txt
```

# 2. Prepare dataset
Download supporting [data](https://drive.google.com/drive/folders/1VcB4z6d0t4rpn5n0N1ZyYQOp5tTQzguA?usp=drive_link).
## 2.1 Download the original dataset
Download TCGA pan-cancer DNA methylation dataset.
```
cd ./data/
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/GDC-PANCAN.methylation450.tsv.gz
```
Download TCGA pan-cancer gene expression dataset.
```
wget https://gdc-hub.s3.us-east-1.amazonaws.com/download/GDC-PANCAN.htseq_fpkm-uq.tsv.gz
```
## 2.2 Preprocess dataset
Return your root path and run the following command.
```
bash prepare_dataset.sh
```
After that, we have two files under `./data/h5/`

# 3. Train AVBAE
## 3.1 Model training
```
bash train_avbae.sh
```
## 3.2 SVM classifier training
```
python -m svm_lf.py
```
# 4. Train AVBAE-MODFR(MODFR w/o pre-trained encoder)
For a fair comparison, we upload our preprocessed sub-datasets and 5-fold dataset splits results.
## 4.1 Model training
```
bash train_modfr.sh
```
## 4.2 Performance on Feature Selection

Test performance on feature importance ranking results obtained by ours.
```
python -m svm_fs.py
```
