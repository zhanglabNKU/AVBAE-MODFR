import torch
from utils import get_sub_dataloader_unsupervised, get_dataloader
from models import AVBAE
from torch.optim import Adam
import pandas as pd
from utils import load_filtered_samples
from args import get_args_avbae

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_avb(epoch, train_dataloader, model, optimizer, optimizer_disc, type='multi'):
    num_batches = 0
    train_recon_loss = 0
    train_disc_loss = 0
    train_loss = 0

    for batch_idx,sample in enumerate(train_dataloader):
        num_batches += 1
        features, _ = sample

        if type == 'multi':
            features = [f.to(device) for f in features]
        elif type == 'methylation':
            features = [f.to(device) for f in features[:-1]]
        elif type == 'expression':
            features = features[-1].to(device)

        loss, disc_loss = model(features)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer.step()
        optimizer_disc.step()

        train_loss += loss.item()
        train_recon_loss += model.loss_recon.item()
        train_disc_loss += disc_loss.item()

    train_loss_ave = train_loss / num_batches
    train_recon_loss_ave = train_recon_loss / num_batches
    train_disc_loss_ave = train_disc_loss / num_batches

    print("Epoch:{:3d},loss:{:.3f},recon_loss:{:.3f},disc_loss:{:.3f}".format(epoch,
                                                                              train_loss_ave,
                                                                              train_recon_loss_ave,
                                                                              train_disc_loss_ave))


@torch.no_grad()
def save_representation(args, full_dataloder, model, filter_datasets=[], save_file='/', type='multi'):
    print('saving latent features')

    samples_id = pd.read_csv(args.data_path + '/samples_id.tsv', sep='\t', index_col=0)
    samples_id = samples_id.filter(items=load_filtered_samples(), axis=0)

    latent_features = []
    model.eval()
    with torch.no_grad():
        for sample in full_dataloder:
            features, label = sample
            if type == 'multi':
                features = [f.to(device) for f in features]
            elif type == 'methylation':
                features = [f.to(device) for f in features[:-1]]
            elif type == 'expression':
                features = features[-1].to(device)
            try:
                mean, _ = model.encode(features)
            except:
                mean = model.encode(features)
            latent_features.append(mean)

    model.train()
    latent_features = torch.cat(latent_features, dim=0)
    latent_features = latent_features.cpu().numpy()
    latent_features = pd.DataFrame(latent_features, index=samples_id.index.values)
    latent_features.to_csv(save_file, sep='\t')


@torch.no_grad()
def save_model(model, file='/model.pt'):
    print('saving model')
    torch.save(model.state_dict(), file)

def main(args):
    dataloader_shuf, dataloader, features_dim_list = get_dataloader(args, pretraining=True)

    print(features_dim_list)
    # filter_datasets = ['TCGA-BRCA'] # 783
    # filter_datasets = ['TCGA-CESC'] # 306
    # filter_datasets = ['TCGA-COAD'] # 308
    # filter_datasets = ['TCGA-ESCA'] # 162
    # filter_datasets = ['TCGA-GBM']  # 63
    # filter_datasets = ['TCGA-HNSC'] # 502
    # filter_datasets = ['TCGA-LGG']  # 529
    # filter_datasets = ['TCGA-READ'] # 99
    # filter_datasets = ['TCGA-SARC'] # 263
    # filter_datasets = ['TCGA-STAD'] # 338
    # filter_datasets = ['TCGA-TGCT'] # 256
    # filter_datasets = ['TCGA-UCEC'] # 433
    # train_dataloader, full_dataloader, features_dim_list = get_sub_dataloader_unsupervised(args, filter_datasets=filter_datasets)

    epochs_pretrain = args.epochs
    hidden_dim_list = [(256, 4096), (1024, 1024), 512]
    model = AVBAE(features_dim_list, hidden_dim_list)
    model = model.to(device)
    # epochs_pretrain = 30
    disc_params = []
    other_params = []
    for name, para in model.named_parameters():
        if 'dis' in name:
            disc_params.append(para)
        else:
            other_params.append(para)
    optimizer = Adam(other_params, lr=args.lr_backbone)
    optimizer_disc = Adam(disc_params, lr=args.lr_dis)

    for epoch in range(1, epochs_pretrain+1):
        train_avb(epoch, dataloader_shuf, model, optimizer, optimizer_disc)

    save_representation_file = args.save_path + 'saved_representation.tsv'
    save_model_file = args.save_path + 'pretrained_model.pt'
    save_representation(args, dataloader, model, save_file=save_representation_file)
    save_model(model, file=save_model_file)


if __name__ == "__main__":
    args = get_args_avbae()
    main(args)