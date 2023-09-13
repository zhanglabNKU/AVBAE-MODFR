import torch
import torch.nn.functional as F
import torch.nn as nn

## 基本VAE结构看一下效果

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.backends.cudnn.benchmark = True
class linear_module(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0, layer_type='1'):
        super(linear_module, self).__init__()
        ## 100 True
        ## all F T F
        if layer_type == '1':
            self.layer = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                       nn.BatchNorm1d(output_dim),
                                       # nn.Dropout(0.5),
                                       nn.ReLU(inplace=True))

        elif layer_type == '0':
            self.layer = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                       nn.BatchNorm1d(output_dim),
                                       nn.Sigmoid())

        elif layer_type == '2':
            self.layer = nn.Sequential(nn.Linear(input_dim, output_dim, bias=False),
                                       nn.BatchNorm1d(output_dim)
                                       )
        elif layer_type == '3':
            self.layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout))


    def forward(self,x):
        o = self.layer(x)
        return o

class MLP_classfier(nn.Module):
    def __init__(self, feature_dim, hidden_dim_list):
        super(MLP_classfier, self).__init__()
        self.encoder_gene_layer0 = linear_module(feature_dim, hidden_dim_list[0])
        self.encoder_gene_layer1 = linear_module(hidden_dim_list[0], hidden_dim_list[1])
        self.encoder_gene_layer2 = linear_module(hidden_dim_list[1], hidden_dim_list[2])
        self.encoder_gene_layer3 = linear_module(hidden_dim_list[2], hidden_dim_list[3])
        self.encoder_gene_layer4 = linear_module(hidden_dim_list[3], hidden_dim_list[-1], layer_type='2')

    def forward(self,x):
        h = self.encoder_gene_layer0(x)
        h = self.encoder_gene_layer1(h)
        h = self.encoder_gene_layer2(h)
        h = self.encoder_gene_layer3(h)
        o = self.encoder_gene_layer4(h)
        return o
## AVBAE
class AVBAE(nn.Module):
    def __init__(self, feature_dim_list, hidden_dim_list, latent_dim=128):
        super(AVBAE, self).__init__()
        self.hidden_dim = hidden_dim_list[0][0]
        self.encoder_chr1 = linear_module(feature_dim_list[0], hidden_dim_list[0][0])

        self.encoder_chr2 = linear_module(feature_dim_list[1], hidden_dim_list[0][0])
        self.encoder_chr3 = linear_module(feature_dim_list[2], hidden_dim_list[0][0])
        self.encoder_chr4 = linear_module(feature_dim_list[3], hidden_dim_list[0][0])
        self.encoder_chr5 = linear_module(feature_dim_list[4], hidden_dim_list[0][0])
        self.encoder_chr6 = linear_module(feature_dim_list[5], hidden_dim_list[0][0])
        self.encoder_chr7 = linear_module(feature_dim_list[6], hidden_dim_list[0][0])
        self.encoder_chr8 = linear_module(feature_dim_list[7], hidden_dim_list[0][0])
        self.encoder_chr9 = linear_module(feature_dim_list[8], hidden_dim_list[0][0])
        self.encoder_chr10 = linear_module(feature_dim_list[9], hidden_dim_list[0][0])
        self.encoder_chr11 = linear_module(feature_dim_list[10], hidden_dim_list[0][0])
        self.encoder_chr12 = linear_module(feature_dim_list[11], hidden_dim_list[0][0])
        self.encoder_chr13 = linear_module(feature_dim_list[12], hidden_dim_list[0][0])
        self.encoder_chr14 = linear_module(feature_dim_list[13], hidden_dim_list[0][0])
        self.encoder_chr15 = linear_module(feature_dim_list[14], hidden_dim_list[0][0])
        self.encoder_chr16 = linear_module(feature_dim_list[15], hidden_dim_list[0][0])
        self.encoder_chr17 = linear_module(feature_dim_list[16], hidden_dim_list[0][0])
        self.encoder_chr18 = linear_module(feature_dim_list[17], hidden_dim_list[0][0])
        self.encoder_chr19 = linear_module(feature_dim_list[18], hidden_dim_list[0][0])
        self.encoder_chr20 = linear_module(feature_dim_list[19], hidden_dim_list[0][0])
        self.encoder_chr21 = linear_module(feature_dim_list[20], hidden_dim_list[0][0])
        self.encoder_chr22 = linear_module(feature_dim_list[21], hidden_dim_list[0][0])
        # self.encoder_chrX = linear_module(feature_dim_list[22], hidden_dim_list[0][0])
        self.encoder_gene_layer0 = linear_module(feature_dim_list[-1], hidden_dim_list[0][1])

        # self.encoder_dna = linear_module(hidden_dim_list[0][0] * 23, hidden_dim_list[1][0])
        self.encoder_dna = linear_module(hidden_dim_list[0][0] * 22, hidden_dim_list[1][0])

        self.encoder_gene_layer1 = linear_module(hidden_dim_list[0][1], hidden_dim_list[1][1])

        self.encoder_integration = linear_module(sum(hidden_dim_list[1]), hidden_dim_list[2])

        self.encoder_mean = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                          nn.BatchNorm1d(latent_dim))

        input_noise_size = latent_dim
        self.input_noise_size = input_noise_size

        self.encoder_logstd = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                            nn.BatchNorm1d(latent_dim))

        self.decoder_layer0 = linear_module(latent_dim, hidden_dim_list[2])
        self.decoder_integration = linear_module(hidden_dim_list[2], sum(hidden_dim_list[1]))

        self.decoder_dna = linear_module(hidden_dim_list[1][0], hidden_dim_list[0][0] * 22)
        self.decoder_gene_layer0 = linear_module(hidden_dim_list[1][1], hidden_dim_list[0][1])

        self.decoder_chr1 = linear_module(hidden_dim_list[0][0], feature_dim_list[0], layer_type='0')  ## LBS
        self.decoder_chr2 = linear_module(hidden_dim_list[0][0], feature_dim_list[1], layer_type='0')
        self.decoder_chr3 = linear_module(hidden_dim_list[0][0], feature_dim_list[2], layer_type='0')
        self.decoder_chr4 = linear_module(hidden_dim_list[0][0], feature_dim_list[3], layer_type='0')
        self.decoder_chr5 = linear_module(hidden_dim_list[0][0], feature_dim_list[4], layer_type='0')
        self.decoder_chr6 = linear_module(hidden_dim_list[0][0], feature_dim_list[5], layer_type='0')
        self.decoder_chr7 = linear_module(hidden_dim_list[0][0], feature_dim_list[6], layer_type='0')
        self.decoder_chr8 = linear_module(hidden_dim_list[0][0], feature_dim_list[7], layer_type='0')
        self.decoder_chr9 = linear_module(hidden_dim_list[0][0], feature_dim_list[8], layer_type='0')
        self.decoder_chr10 = linear_module(hidden_dim_list[0][0], feature_dim_list[9], layer_type='0')
        self.decoder_chr11 = linear_module(hidden_dim_list[0][0], feature_dim_list[10], layer_type='0')
        self.decoder_chr12 = linear_module(hidden_dim_list[0][0], feature_dim_list[11], layer_type='0')
        self.decoder_chr13 = linear_module(hidden_dim_list[0][0], feature_dim_list[12], layer_type='0')
        self.decoder_chr14 = linear_module(hidden_dim_list[0][0], feature_dim_list[13], layer_type='0')
        self.decoder_chr15 = linear_module(hidden_dim_list[0][0], feature_dim_list[14], layer_type='0')
        self.decoder_chr16 = linear_module(hidden_dim_list[0][0], feature_dim_list[15], layer_type='0')
        self.decoder_chr17 = linear_module(hidden_dim_list[0][0], feature_dim_list[16], layer_type='0')
        self.decoder_chr18 = linear_module(hidden_dim_list[0][0], feature_dim_list[17], layer_type='0')
        self.decoder_chr19 = linear_module(hidden_dim_list[0][0], feature_dim_list[18], layer_type='0')
        self.decoder_chr20 = linear_module(hidden_dim_list[0][0], feature_dim_list[19], layer_type='0')
        self.decoder_chr21 = linear_module(hidden_dim_list[0][0], feature_dim_list[20], layer_type='0')
        self.decoder_chr22 = linear_module(hidden_dim_list[0][0], feature_dim_list[21], layer_type='0')
        # self.decoder_chrX = linear_module(hidden_dim_list[0][0], feature_dim_list[22], layer_type='0')
        self.decoder_gene_layer1 = linear_module(hidden_dim_list[0][1], feature_dim_list[-1], layer_type='0')

        self.dis_recon = linear_module(latent_dim, 512, layer_type='3') ## LRD
        self.dis_layer0 = linear_module(512*2, 512, layer_type='3')
        self.dis_layer1 = linear_module(512, 256, layer_type='3')
        self.dis_layer2 = linear_module(256, 128, layer_type='3')
        self.dis_layer3 = nn.Linear(128, 1)

    def encode(self, x):
        hidden_chr1 = self.encoder_chr1(x[0])
        hidden_chr2 = self.encoder_chr2(x[1])
        hidden_chr3 = self.encoder_chr3(x[2])
        hidden_chr4 = self.encoder_chr4(x[3])
        hidden_chr5 = self.encoder_chr5(x[4])
        hidden_chr6 = self.encoder_chr6(x[5])
        hidden_chr7 = self.encoder_chr7(x[6])
        hidden_chr8 = self.encoder_chr8(x[7])
        hidden_chr9 = self.encoder_chr9(x[8])
        hidden_chr10 = self.encoder_chr10(x[9])
        hidden_chr11 = self.encoder_chr11(x[10])
        hidden_chr12 = self.encoder_chr12(x[11])
        hidden_chr13 = self.encoder_chr13(x[12])
        hidden_chr14 = self.encoder_chr14(x[13])
        hidden_chr15 = self.encoder_chr15(x[14])
        hidden_chr16 = self.encoder_chr16(x[15])
        hidden_chr17 = self.encoder_chr17(x[16])
        hidden_chr18 = self.encoder_chr18(x[17])
        hidden_chr19 = self.encoder_chr19(x[18])
        hidden_chr20 = self.encoder_chr20(x[19])
        hidden_chr21 = self.encoder_chr21(x[20])
        hidden_chr22 = self.encoder_chr22(x[21])
        hidden_gene0 = self.encoder_gene_layer0(x[-1])

        self.hidden_features = [hidden_chr1,hidden_chr2,hidden_chr3,
                                                 hidden_chr4,hidden_chr5,hidden_chr6,
                                                 hidden_chr7,hidden_chr8,hidden_chr9,
                                                 hidden_chr10,hidden_chr11,hidden_chr12,
                                                 hidden_chr13,hidden_chr14,hidden_chr15,
                                                 hidden_chr16,hidden_chr17,hidden_chr18,
                                                 hidden_chr19,hidden_chr20,hidden_chr21,
                                                 hidden_chr22,
                                                 hidden_gene0
                                                 ]

        hidden_dna = self.encoder_dna(torch.cat([hidden_chr1,hidden_chr2,hidden_chr3,
                                                 hidden_chr4,hidden_chr5,hidden_chr6,
                                                 hidden_chr7,hidden_chr8,hidden_chr9,
                                                 hidden_chr10,hidden_chr11,hidden_chr12,
                                                 hidden_chr13,hidden_chr14,hidden_chr15,
                                                 hidden_chr16,hidden_chr17,hidden_chr18,
                                                 hidden_chr19,hidden_chr20,hidden_chr21,
                                                 hidden_chr22,
                                                 ], dim=1))


        hidden_gene1 = self.encoder_gene_layer1(hidden_gene0)
        hidden_integrated = self.encoder_integration(torch.cat([hidden_dna, hidden_gene1], dim=1))

        self.hidden_integrated = hidden_integrated

        mean = self.encoder_mean(hidden_integrated)
        logstd = self.encoder_logstd(hidden_integrated)

        return mean, logstd

    def decode(self, z):
        hidden = self.decoder_layer0(z)
        hidden_integrated = self.decoder_integration(hidden)
        hidden_dna = self.decoder_dna(hidden_integrated[:,:hidden_integrated.shape[1]//2])
        hidden_gene = self.decoder_gene_layer0(hidden_integrated[:,hidden_integrated.shape[1]//2:])
        recon_gene = self.decoder_gene_layer1(hidden_gene)
        hidden_dna_list = []
        hidden_dim = self.hidden_dim

        for i in range(1, 23):
            hidden_dna_list.append(hidden_dna[:, (i-1)*hidden_dim:i*hidden_dim])

        recon_chr1 = self.decoder_chr1(hidden_dna_list[0])
        recon_chr2 = self.decoder_chr2(hidden_dna_list[1])
        recon_chr3 = self.decoder_chr3(hidden_dna_list[2])
        recon_chr4 = self.decoder_chr4(hidden_dna_list[3])
        recon_chr5 = self.decoder_chr5(hidden_dna_list[4])
        recon_chr6 = self.decoder_chr6(hidden_dna_list[5])
        recon_chr7 = self.decoder_chr7(hidden_dna_list[6])
        recon_chr8 = self.decoder_chr8(hidden_dna_list[7])
        recon_chr9 = self.decoder_chr9(hidden_dna_list[8])
        recon_chr10 = self.decoder_chr10(hidden_dna_list[9])
        recon_chr11 = self.decoder_chr11(hidden_dna_list[10])
        recon_chr12 = self.decoder_chr12(hidden_dna_list[11])
        recon_chr13 = self.decoder_chr13(hidden_dna_list[12])
        recon_chr14 = self.decoder_chr14(hidden_dna_list[13])
        recon_chr15 = self.decoder_chr15(hidden_dna_list[14])
        recon_chr16 = self.decoder_chr16(hidden_dna_list[15])
        recon_chr17 = self.decoder_chr17(hidden_dna_list[16])
        recon_chr18 = self.decoder_chr18(hidden_dna_list[17])
        recon_chr19 = self.decoder_chr19(hidden_dna_list[18])
        recon_chr20 = self.decoder_chr20(hidden_dna_list[19])
        recon_chr21 = self.decoder_chr21(hidden_dna_list[20])
        recon_chr22 = self.decoder_chr22(hidden_dna_list[21])


        recon_features_list = [recon_chr1,recon_chr2,recon_chr3,
                               recon_chr4,recon_chr5,recon_chr6,
                               recon_chr7,recon_chr8,recon_chr9,
                               recon_chr10,recon_chr11,recon_chr12,
                               recon_chr13,recon_chr14,recon_chr15,
                               recon_chr16,recon_chr17,recon_chr18,
                               recon_chr19,recon_chr20,recon_chr21,
                               recon_chr22,
                               recon_gene]
        return recon_features_list

    def sample_prior(self, z):
        p = torch.randn_like(z, device=device)
        return p

    def sample_posterior(self, mean, logstd=1):
        gaussian_noise = torch.randn_like(mean, device=device)
        p = mean + gaussian_noise * torch.exp(logstd)
        return p

    def discriminator(self, mean, z):
        recon = self.dis_recon(z)
        h = self.dis_layer0(torch.cat([mean, recon], dim=1))
        h = self.dis_layer1(h)
        h = self.dis_layer2(h)
        h = self.dis_layer3(h)
        return h

    def recon_loss(self, x_list, recon_x_list):
        loss = 0
        for x, recon_x in zip(x_list[:-1], recon_x_list[:-1]):
            cur_loss = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).mean() ## 'sum'
            # cur_loss = F.binary_cross_entropy(recon_x, x, reduction='none').mean()
            loss += cur_loss
        # loss = loss / 23
        loss = loss / 22
        loss = loss + F.binary_cross_entropy(recon_x_list[-1], x_list[-1], reduction='none').sum(1).mean()
        # loss = loss + F.binary_cross_entropy(recon_x_list[-1], x_list[-1], reduction='none').mean()
        return loss


    def forward(self, x_list):

        mean, logstd = self.encode(x_list)
        self.mean = mean

        z_q = self.sample_posterior(mean, logstd)
        z_p = self.sample_prior(z_q)

        log_posterior = self.discriminator(self.hidden_integrated, z_q)
        log_prior_dis = self.discriminator(self.hidden_integrated.detach(), z_p)
        log_posterior_dis = self.discriminator(self.hidden_integrated.detach(), z_q.detach())

        disc_loss = torch.sum(F.binary_cross_entropy_with_logits(log_posterior_dis, torch.ones_like(log_posterior_dis),
                                                                 reduction='none').sum(1).mean()
                              + F.binary_cross_entropy_with_logits(log_prior_dis, torch.zeros_like(log_prior_dis),
                                                                   reduction='none').sum(1).mean())

        kl = log_posterior.mean()
        recon_list = self.decode(z_q)
        loss_recon = self.recon_loss(x_list, recon_list)
        self.loss_recon = loss_recon
        loss = loss_recon + kl
        return loss, disc_loss


class AVBAE_Methylation(nn.Module):
    def __init__(self, feature_dim_list):
        super(AVBAE_Methylation, self).__init__()
        # hidden_dim_list = [(256, 4096), (1024, 1024), 512]
        hidden_dim_list = [256, 1024, 512]
        self.hidden_dim = hidden_dim_list[0]
        latent_dim = 128
        self.encoder_chr1 = linear_module(feature_dim_list[0], hidden_dim_list[0])
        self.encoder_chr2 = linear_module(feature_dim_list[1], hidden_dim_list[0])
        self.encoder_chr3 = linear_module(feature_dim_list[2], hidden_dim_list[0])
        self.encoder_chr4 = linear_module(feature_dim_list[3], hidden_dim_list[0])
        self.encoder_chr5 = linear_module(feature_dim_list[4], hidden_dim_list[0])
        self.encoder_chr6 = linear_module(feature_dim_list[5], hidden_dim_list[0])
        self.encoder_chr7 = linear_module(feature_dim_list[6], hidden_dim_list[0])
        self.encoder_chr8 = linear_module(feature_dim_list[7], hidden_dim_list[0])
        self.encoder_chr9 = linear_module(feature_dim_list[8], hidden_dim_list[0])
        self.encoder_chr10 = linear_module(feature_dim_list[9], hidden_dim_list[0])
        self.encoder_chr11 = linear_module(feature_dim_list[10], hidden_dim_list[0])
        self.encoder_chr12 = linear_module(feature_dim_list[11], hidden_dim_list[0])
        self.encoder_chr13 = linear_module(feature_dim_list[12], hidden_dim_list[0])
        self.encoder_chr14 = linear_module(feature_dim_list[13], hidden_dim_list[0])
        self.encoder_chr15 = linear_module(feature_dim_list[14], hidden_dim_list[0])
        self.encoder_chr16 = linear_module(feature_dim_list[15], hidden_dim_list[0])
        self.encoder_chr17 = linear_module(feature_dim_list[16], hidden_dim_list[0])
        self.encoder_chr18 = linear_module(feature_dim_list[17], hidden_dim_list[0])
        self.encoder_chr19 = linear_module(feature_dim_list[18], hidden_dim_list[0])
        self.encoder_chr20 = linear_module(feature_dim_list[19], hidden_dim_list[0])
        self.encoder_chr21 = linear_module(feature_dim_list[20], hidden_dim_list[0])
        self.encoder_chr22 = linear_module(feature_dim_list[21], hidden_dim_list[0])

        self.encoder_dna = linear_module(hidden_dim_list[0] * 22, hidden_dim_list[1])
        self.encoder_integration = linear_module(hidden_dim_list[1], hidden_dim_list[2])

        self.encoder_mean = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                          nn.BatchNorm1d(latent_dim))

        self.encoder_logstd = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                            nn.BatchNorm1d(latent_dim))

        self.decoder_layer0 = linear_module(latent_dim, hidden_dim_list[2])
        self.decoder_integration = linear_module(hidden_dim_list[2], hidden_dim_list[1])

        self.decoder_dna = linear_module(hidden_dim_list[1], hidden_dim_list[0] * 22)

        self.decoder_chr1 = linear_module(hidden_dim_list[0], feature_dim_list[0], layer_type='0')  ## LBS
        self.decoder_chr2 = linear_module(hidden_dim_list[0], feature_dim_list[1], layer_type='0')
        self.decoder_chr3 = linear_module(hidden_dim_list[0], feature_dim_list[2], layer_type='0')
        self.decoder_chr4 = linear_module(hidden_dim_list[0], feature_dim_list[3], layer_type='0')
        self.decoder_chr5 = linear_module(hidden_dim_list[0], feature_dim_list[4], layer_type='0')
        self.decoder_chr6 = linear_module(hidden_dim_list[0], feature_dim_list[5], layer_type='0')
        self.decoder_chr7 = linear_module(hidden_dim_list[0], feature_dim_list[6], layer_type='0')
        self.decoder_chr8 = linear_module(hidden_dim_list[0], feature_dim_list[7], layer_type='0')
        self.decoder_chr9 = linear_module(hidden_dim_list[0], feature_dim_list[8], layer_type='0')
        self.decoder_chr10 = linear_module(hidden_dim_list[0], feature_dim_list[9], layer_type='0')
        self.decoder_chr11 = linear_module(hidden_dim_list[0], feature_dim_list[10], layer_type='0')
        self.decoder_chr12 = linear_module(hidden_dim_list[0], feature_dim_list[11], layer_type='0')
        self.decoder_chr13 = linear_module(hidden_dim_list[0], feature_dim_list[12], layer_type='0')
        self.decoder_chr14 = linear_module(hidden_dim_list[0], feature_dim_list[13], layer_type='0')
        self.decoder_chr15 = linear_module(hidden_dim_list[0], feature_dim_list[14], layer_type='0')
        self.decoder_chr16 = linear_module(hidden_dim_list[0], feature_dim_list[15], layer_type='0')
        self.decoder_chr17 = linear_module(hidden_dim_list[0], feature_dim_list[16], layer_type='0')
        self.decoder_chr18 = linear_module(hidden_dim_list[0], feature_dim_list[17], layer_type='0')
        self.decoder_chr19 = linear_module(hidden_dim_list[0], feature_dim_list[18], layer_type='0')
        self.decoder_chr20 = linear_module(hidden_dim_list[0], feature_dim_list[19], layer_type='0')
        self.decoder_chr21 = linear_module(hidden_dim_list[0], feature_dim_list[20], layer_type='0')
        self.decoder_chr22 = linear_module(hidden_dim_list[0], feature_dim_list[21], layer_type='0')

        input_noise_size = latent_dim
        self.input_noise_size = input_noise_size

        self.dis_recon = linear_module(latent_dim, 512, layer_type='3')  ## LRD
        self.dis_layer0 = linear_module(512 * 2, 512, layer_type='3')
        self.dis_layer1 = linear_module(512, 256, layer_type='3')
        self.dis_layer2 = linear_module(256, 128, layer_type='3')
        self.dis_layer3 = nn.Linear(128, 1)

    def encode(self, x):
        hidden_chr1 = self.encoder_chr1(x[0])
        hidden_chr2 = self.encoder_chr2(x[1])
        hidden_chr3 = self.encoder_chr3(x[2])
        hidden_chr4 = self.encoder_chr4(x[3])
        hidden_chr5 = self.encoder_chr5(x[4])
        hidden_chr6 = self.encoder_chr6(x[5])
        hidden_chr7 = self.encoder_chr7(x[6])
        hidden_chr8 = self.encoder_chr8(x[7])
        hidden_chr9 = self.encoder_chr9(x[8])
        hidden_chr10 = self.encoder_chr10(x[9])
        hidden_chr11 = self.encoder_chr11(x[10])
        hidden_chr12 = self.encoder_chr12(x[11])
        hidden_chr13 = self.encoder_chr13(x[12])
        hidden_chr14 = self.encoder_chr14(x[13])
        hidden_chr15 = self.encoder_chr15(x[14])
        hidden_chr16 = self.encoder_chr16(x[15])
        hidden_chr17 = self.encoder_chr17(x[16])
        hidden_chr18 = self.encoder_chr18(x[17])
        hidden_chr19 = self.encoder_chr19(x[18])
        hidden_chr20 = self.encoder_chr20(x[19])
        hidden_chr21 = self.encoder_chr21(x[20])
        hidden_chr22 = self.encoder_chr22(x[21])

        hidden_dna = self.encoder_dna(torch.cat([hidden_chr1, hidden_chr2, hidden_chr3,
                                                 hidden_chr4, hidden_chr5, hidden_chr6,
                                                 hidden_chr7, hidden_chr8, hidden_chr9,
                                                 hidden_chr10, hidden_chr11, hidden_chr12,
                                                 hidden_chr13, hidden_chr14, hidden_chr15,
                                                 hidden_chr16, hidden_chr17, hidden_chr18,
                                                 hidden_chr19, hidden_chr20, hidden_chr21,
                                                 hidden_chr22
                                                 ], dim=1))

        hidden_integrated = self.encoder_integration(hidden_dna)
        self.hidden_integrated = hidden_integrated
        mean = self.encoder_mean(hidden_integrated)
        logstd = self.encoder_logstd(hidden_integrated)
        return mean, logstd

    def decode(self, z):
        hidden = self.decoder_layer0(z)
        hidden_integrated = self.decoder_integration(hidden)

        hidden_dna = self.decoder_dna(hidden_integrated)
        hidden_dna_list = []
        hidden_dim = self.hidden_dim
        for i in range(1, 23):
            hidden_dna_list.append(hidden_dna[:, (i - 1) * hidden_dim:i * hidden_dim])
        recon_chr1 = self.decoder_chr1(hidden_dna_list[0])
        recon_chr2 = self.decoder_chr2(hidden_dna_list[1])
        recon_chr3 = self.decoder_chr3(hidden_dna_list[2])
        recon_chr4 = self.decoder_chr4(hidden_dna_list[3])
        recon_chr5 = self.decoder_chr5(hidden_dna_list[4])
        recon_chr6 = self.decoder_chr6(hidden_dna_list[5])
        recon_chr7 = self.decoder_chr7(hidden_dna_list[6])
        recon_chr8 = self.decoder_chr8(hidden_dna_list[7])
        recon_chr9 = self.decoder_chr9(hidden_dna_list[8])
        recon_chr10 = self.decoder_chr10(hidden_dna_list[9])
        recon_chr11 = self.decoder_chr11(hidden_dna_list[10])
        recon_chr12 = self.decoder_chr12(hidden_dna_list[11])
        recon_chr13 = self.decoder_chr13(hidden_dna_list[12])
        recon_chr14 = self.decoder_chr14(hidden_dna_list[13])
        recon_chr15 = self.decoder_chr15(hidden_dna_list[14])
        recon_chr16 = self.decoder_chr16(hidden_dna_list[15])
        recon_chr17 = self.decoder_chr17(hidden_dna_list[16])
        recon_chr18 = self.decoder_chr18(hidden_dna_list[17])
        recon_chr19 = self.decoder_chr19(hidden_dna_list[18])
        recon_chr20 = self.decoder_chr20(hidden_dna_list[19])
        recon_chr21 = self.decoder_chr21(hidden_dna_list[20])
        recon_chr22 = self.decoder_chr22(hidden_dna_list[21])

        recon_features_list = [recon_chr1, recon_chr2, recon_chr3,
                               recon_chr4, recon_chr5, recon_chr6,
                               recon_chr7, recon_chr8, recon_chr9,
                               recon_chr10, recon_chr11, recon_chr12,
                               recon_chr13, recon_chr14, recon_chr15,
                               recon_chr16, recon_chr17, recon_chr18,
                               recon_chr19, recon_chr20, recon_chr21,
                               recon_chr22
                               ]
        return recon_features_list

    def sample_prior(self, z):
        p = torch.randn_like(z, device=device)
        return p

    def sample_posterior(self, mean, logstd=1):
        gaussian_noise = torch.randn_like(mean, device=device)
        p = mean + gaussian_noise * torch.exp(logstd)
        return p

    def discriminator(self, mean, z):
        recon = self.dis_recon(z)
        h = self.dis_layer0(torch.cat([mean, recon], dim=1))
        h = self.dis_layer1(h)
        h = self.dis_layer2(h)
        h = self.dis_layer3(h)
        return h

    def recon_loss(self, x_list, recon_x_list):
        loss = 0
        for x, recon_x in zip(x_list, recon_x_list):
            cur_loss = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).mean()
            loss += cur_loss
        loss = loss / 22
        return loss

    def forward(self, x_list):
        mean, logstd = self.encode(x_list)
        self.mean = mean
        z_q = self.sample_posterior(mean, logstd)
        z_p = self.sample_prior(z_q)

        log_posterior = self.discriminator(self.hidden_integrated, z_q)
        log_prior_dis = self.discriminator(self.hidden_integrated.detach(), z_p)
        log_posterior_dis = self.discriminator(self.hidden_integrated.detach(), z_q.detach())

        disc_loss = torch.sum(F.binary_cross_entropy_with_logits(log_posterior_dis, torch.ones_like(log_posterior_dis),
                                                                 reduction='none').sum(1).mean()
                              + F.binary_cross_entropy_with_logits(log_prior_dis, torch.zeros_like(log_prior_dis),
                                                                   reduction='none').sum(1).mean())

        kl = log_posterior.mean()
        recon_list = self.decode(z_q)
        loss_recon = self.recon_loss(x_list, recon_list)
        self.loss_recon = loss_recon
        loss = loss_recon + kl
        return loss, disc_loss

class AVBAE_Expression(nn.Module):
    def __init__(self, feature_dim):
        super(AVBAE_Expression, self).__init__()
        hidden_dim_list = [4096, 1024, 512]
        latent_dim = 128
        self.encoder_gene_layer0 = linear_module(feature_dim, hidden_dim_list[0])
        self.encoder_gene_layer1 = linear_module(hidden_dim_list[0], hidden_dim_list[1])
        self.encoder_integration = linear_module(hidden_dim_list[1], hidden_dim_list[2])
        ## 先这样
        self.encoder_mean = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                          nn.BatchNorm1d(latent_dim))
        self.encoder_logstd = nn.Sequential(nn.Linear(hidden_dim_list[2], latent_dim, bias=False),
                                            nn.BatchNorm1d(latent_dim))
        self.decoder_layer0 = linear_module(latent_dim, hidden_dim_list[2])
        self.decoder_integration = linear_module(hidden_dim_list[2], hidden_dim_list[1])
        self.decoder_gene_layer0 = linear_module(hidden_dim_list[1], hidden_dim_list[0])
        self.decoder_gene_layer1 = linear_module(hidden_dim_list[0], feature_dim, layer_type='0')

        input_noise_size = latent_dim
        self.input_noise_size = input_noise_size

        self.dis_recon = linear_module(latent_dim, 512, layer_type='3')  ## LRD
        self.dis_layer0 = linear_module(512 * 2, 512, layer_type='3')
        self.dis_layer1 = linear_module(512, 256, layer_type='3')
        self.dis_layer2 = linear_module(256, 128, layer_type='3')
        self.dis_layer3 = nn.Linear(128, 1)

    def encode(self, x):  ## x is not a list
        hidden_gene0 = self.encoder_gene_layer0(x)
        hidden_gene1 = self.encoder_gene_layer1(hidden_gene0)
        hidden_integrated = self.encoder_integration(hidden_gene1)
        self.hidden_integrated = hidden_integrated

        mean = self.encoder_mean(hidden_integrated)
        logstd = self.encoder_logstd(hidden_integrated)
        return mean, logstd

    def decode(self, z):
        hidden = self.decoder_layer0(z)
        hidden_integrated = self.decoder_integration(hidden)
        hidden_gene = self.decoder_gene_layer0(hidden_integrated)
        recon_gene = self.decoder_gene_layer1(hidden_gene)
        return recon_gene

    def sample_prior(self, z):
        p = torch.randn_like(z, device=device)
        return p

    def sample_posterior(self, mean, logstd=1):
        gaussian_noise = torch.randn_like(mean, device=device)
        p = mean + gaussian_noise * torch.exp(logstd)
        return p

    def discriminator(self, mean, z):
        recon = self.dis_recon(z)
        h = self.dis_layer0(torch.cat([mean, recon], dim=1))
        h = self.dis_layer1(h)
        h = self.dis_layer2(h)
        h = self.dis_layer3(h)
        return h

    def recon_loss(self, x, recon_x):
        loss = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).mean()
        return loss

    def forward(self, x):
        mean, logstd = self.encode(x)
        self.mean = mean
        z_q = self.sample_posterior(mean, logstd)
        z_p = self.sample_prior(z_q)

        log_posterior = self.discriminator(self.hidden_integrated, z_q)
        log_prior_dis = self.discriminator(self.hidden_integrated.detach(), z_p)
        log_posterior_dis = self.discriminator(self.hidden_integrated.detach(), z_q.detach())

        disc_loss = torch.sum(F.binary_cross_entropy_with_logits(log_posterior_dis, torch.ones_like(log_posterior_dis),
                                                                 reduction='none').sum(1).mean()
                              + F.binary_cross_entropy_with_logits(log_prior_dis, torch.zeros_like(log_prior_dis),
                                                                   reduction='none').sum(1).mean())
        kl = log_posterior.mean()
        recon_x = self.decode(z_q)
        loss_recon = self.recon_loss(x, recon_x)
        self.loss_recon = loss_recon
        loss = loss_recon + kl
        return loss, disc_loss

class classifier(nn.Module):
    def __init__(self, latent_dim, label_dim=2):
        super(classifier, self).__init__()
        self.classifier_layer0 = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                               nn.BatchNorm1d(latent_dim),
                                               nn.ReLU(),
                                               # nn.Dropout(p=0.5)
                                               )
        self.classifier_layer1 = nn.Sequential(nn.Linear(latent_dim, latent_dim//2),
                                               nn.BatchNorm1d(latent_dim//2),
                                               nn.ReLU(),
                                               # nn.Dropout(p=0.5)
                                               )
        self.claasifier_layer2 = nn.Sequential(nn.Linear(latent_dim // 2, label_dim),
                                               nn.BatchNorm1d(label_dim),
                                               )

        # self.classifier_layer0 = nn.Sequential(nn.Linear(latent_dim, latent_dim//2),
        #                                        nn.BatchNorm1d(latent_dim//2),
        #                                        nn.ReLU(),
        #                                        # nn.Dropout(p=0.5)
        #                                        )
        # self.classifier_layer1 = nn.Sequential(nn.Linear(latent_dim//2, label_dim),
        #                                        nn.BatchNorm1d(label_dim),
        #                                        # nn.Dropout(p=0.5)
        #                                        )

    def forward(self, x):
        ##
        h0 = self.classifier_layer0(x)
        h0 = self.classifier_layer1(h0)
        o = self.claasifier_layer2(h0)
        ##
        # h0 = self.classifier_layer0(x)
        # o = self.classifier_layer1(h0)
        return o




















'''
class fs_vae_semi_supervised(nn.Module):
    def __init__(self, feature_dim_list, hidden_dim_list):
        super(fs_vae_semi_supervised, self).__init__()

        # self.vae_model = VAE_avb(feature_dim_list,hidden_dim_list,type='None')
        self.vae_model = VAE_avb_fs(feature_dim_list, hidden_dim_list)
        unmasked_size_mrna = feature_dim_list[-1]//2
        unmasked_size_dnam = [f//4 for f in feature_dim_list]
        # unmasked_size_dnam = [2048]*23
        # unmasked_size_mrna = 8192
        # unmasked_size_dnam = [50]*23
        # unmasked_size_mrna = 50
        mask_size = 16
        self.mask_size = mask_size
        self.classifier = classifier(128)
        ## FIR
        ## 不能选出好的特征　问题还是在这里
        self.fs_chr1 = FeatureSelector_FIR(feature_dim_list[0], unmasked_size_dnam[0], mask_size)
        self.fs_chr2 = FeatureSelector_FIR(feature_dim_list[1], unmasked_size_dnam[1], mask_size)
        self.fs_chr3 = FeatureSelector_FIR(feature_dim_list[2], unmasked_size_dnam[2], mask_size)
        self.fs_chr4 = FeatureSelector_FIR(feature_dim_list[3], unmasked_size_dnam[3], mask_size)
        self.fs_chr5 = FeatureSelector_FIR(feature_dim_list[4], unmasked_size_dnam[4], mask_size)
        self.fs_chr6 = FeatureSelector_FIR(feature_dim_list[5], unmasked_size_dnam[5], mask_size)
        self.fs_chr7 = FeatureSelector_FIR(feature_dim_list[6], unmasked_size_dnam[6], mask_size)
        self.fs_chr8 = FeatureSelector_FIR(feature_dim_list[7], unmasked_size_dnam[7], mask_size)
        self.fs_chr9 = FeatureSelector_FIR(feature_dim_list[8], unmasked_size_dnam[8], mask_size)
        self.fs_chr10 = FeatureSelector_FIR(feature_dim_list[9], unmasked_size_dnam[9], mask_size)
        self.fs_chr11 = FeatureSelector_FIR(feature_dim_list[10], unmasked_size_dnam[10], mask_size)
        self.fs_chr12 = FeatureSelector_FIR(feature_dim_list[11], unmasked_size_dnam[11], mask_size)
        self.fs_chr13 = FeatureSelector_FIR(feature_dim_list[12], unmasked_size_dnam[12], mask_size)
        self.fs_chr14 = FeatureSelector_FIR(feature_dim_list[13], unmasked_size_dnam[13], mask_size)
        self.fs_chr15 = FeatureSelector_FIR(feature_dim_list[14], unmasked_size_dnam[14], mask_size)
        self.fs_chr16 = FeatureSelector_FIR(feature_dim_list[15], unmasked_size_dnam[15], mask_size)
        self.fs_chr17 = FeatureSelector_FIR(feature_dim_list[16], unmasked_size_dnam[16], mask_size)
        self.fs_chr18 = FeatureSelector_FIR(feature_dim_list[17], unmasked_size_dnam[17], mask_size)
        self.fs_chr19 = FeatureSelector_FIR(feature_dim_list[18], unmasked_size_dnam[18], mask_size)
        self.fs_chr20 = FeatureSelector_FIR(feature_dim_list[19], unmasked_size_dnam[19], mask_size)
        self.fs_chr21 = FeatureSelector_FIR(feature_dim_list[20], unmasked_size_dnam[20], mask_size)
        self.fs_chr22 = FeatureSelector_FIR(feature_dim_list[21], unmasked_size_dnam[21], mask_size)
        self.fs_chrX = FeatureSelector_FIR(feature_dim_list[22], unmasked_size_dnam[22], mask_size)
        self.fs_gene = FeatureSelector_FIR(feature_dim_list[23], unmasked_size_mrna, mask_size)

        ## GA
        # self.fs_chr1 = FeatureSelector_genetic(feature_dim_list[0], mask_size)
        # self.fs_chr2 = FeatureSelector_genetic(feature_dim_list[1], mask_size)
        # self.fs_chr3 = FeatureSelector_genetic(feature_dim_list[2], mask_size)
        # self.fs_chr4 = FeatureSelector_genetic(feature_dim_list[3], mask_size)
        # self.fs_chr5 = FeatureSelector_genetic(feature_dim_list[4], mask_size)
        # self.fs_chr6 = FeatureSelector_genetic(feature_dim_list[5], mask_size)
        # self.fs_chr7 = FeatureSelector_genetic(feature_dim_list[6], mask_size)
        # self.fs_chr8 = FeatureSelector_genetic(feature_dim_list[7], mask_size)
        # self.fs_chr9 = FeatureSelector_genetic(feature_dim_list[8], mask_size)
        # self.fs_chr10 = FeatureSelector_genetic(feature_dim_list[9], mask_size)
        # self.fs_chr11 = FeatureSelector_genetic(feature_dim_list[10], mask_size)
        # self.fs_chr12 = FeatureSelector_genetic(feature_dim_list[11], mask_size)
        # self.fs_chr13 = FeatureSelector_genetic(feature_dim_list[12], mask_size)
        # self.fs_chr14 = FeatureSelector_genetic(feature_dim_list[13], mask_size)
        # self.fs_chr15 = FeatureSelector_genetic(feature_dim_list[14], mask_size)
        # self.fs_chr16 = FeatureSelector_genetic(feature_dim_list[15], mask_size)
        # self.fs_chr17 = FeatureSelector_genetic(feature_dim_list[16], mask_size)
        # self.fs_chr18 = FeatureSelector_genetic(feature_dim_list[17], mask_size)
        # self.fs_chr19 = FeatureSelector_genetic(feature_dim_list[18], mask_size)
        # self.fs_chr20 = FeatureSelector_genetic(feature_dim_list[19], mask_size)
        # self.fs_chr21 = FeatureSelector_genetic(feature_dim_list[20], mask_size)
        # self.fs_chr22 = FeatureSelector_genetic(feature_dim_list[21], mask_size)
        # self.fs_chrX = FeatureSelector_genetic(feature_dim_list[22], mask_size)
        # self.fs_gene = FeatureSelector_genetic(feature_dim_list[23], mask_size)

        self.fs = [self.fs_chr1, self.fs_chr2, self.fs_chr3, self.fs_chr4, self.fs_chr5,
                   self.fs_chr6, self.fs_chr7, self.fs_chr8, self.fs_chr9, self.fs_chr10,
                   self.fs_chr11, self.fs_chr12, self.fs_chr13, self.fs_chr14, self.fs_chr15,
                   self.fs_chr16, self.fs_chr17, self.fs_chr18, self.fs_chr19, self.fs_chr20,
                   self.fs_chr21, self.fs_chr22, self.fs_chrX, self.fs_gene]

        self.e = 0
        self.num_batch_pretrain_sel = 4000 ## 5

    def init_opt(self):
        for fs in self.fs:
            fs.set_optimizer(0.001)
        disc_params = []
        vae_params = []
        for name,para in self.vae_model.named_parameters():
            if 'dis' in name:
                disc_params.append(para)
            else:
                vae_params.append(para)

        self.opt_vae = Adam(vae_params, lr=0.001)
        self.opt_disc = Adam(disc_params, lr=0.001)
        self.opt_clf = Adam(self.classifier.parameters(), lr=0.001)

        complem_param = []

        for fs in self.fs:
            for name, para in fs.named_parameters():
                if 'com' in name:
                    complem_param.append(para)

        # print(complem_param)
        self.opt_cpl = Adam(complem_param, lr=0.005)

    def train_model(self, x, y, pretrain=True):
        ## train_fs
        ## version-1
        # for i in range(24):
        #     # print('training selector:{}'.format(i))
        #     fs = self.fs[i]
        #     m_cur = fs.get_m()
        #     x_cur,y_cur = fs.generate_train_data(x[i],m_cur,y)
        #     try:
        #         x_seg[i] = x_cur
        #     except:
        #         x_seg = [torch.zeros_like(x_cur, device=device) for _ in range(len(x))]
        #         x_seg[i] = x_cur
        #
        #     mean, _, _= self.vae_model.encode(x_seg)
        #
        #     x_seg[i] = torch.zeros_like(x_cur,device=device)
        #     pred = self.classifier(mean)
        #     loss_sel = F.cross_entropy(pred,y_cur,reduction='none')
        #     fs.train_sel(m_cur,loss_sel)
        #
        # ## train_clf
        # # print('training classifier encoder decoder')
        # for i in range(24):
        #     fs = self.fs[i]
        #     m_cur = fs.last_best_m_opt
        #     x_cur = x[i]*m_cur
        #     x[i] = x_cur
        #
        # loss_vae, disc_loss = self.vae_model(x,y,classify=False)
        # mean = self.vae_model.mean
        # self.pred = self.classifier(mean)
        #
        # loss_clf = F.cross_entropy(self.pred,y,reduction='mean')
        #
        # loss = loss_clf + loss_vae
        # self.opt_vae.zero_grad()
        # loss.backward(retain_graph = True)
        # self.opt_disc.zero_grad()
        # disc_loss.backward()
        # self.opt_vae.step()
        # self.opt_disc.step()
        #
        # self.loss_clf = loss_clf
        # self.disc_loss = disc_loss
        # self.loss_vae = loss_vae

        ## version-2
        ## pretrain_vae
        if pretrain:
        # if True:
            ## 处理特征
            # t = time.time()
            x_mask = []
            for i in range(24):
                # print('training selector:{}'.format(i))
                fs = self.fs[i]
                m_cur = fs.get_m(pretrain=True)
                # print(m_cur.device)
                x_cur, _ = fs.generate_train_data(x[i], m_cur.to(device), y)
                x_mask.append(x_cur)
                # try:
                #     x_mask[i] = x_cur
                # except:
                #     x_mask = [torch.zeros_like(x_cur, device=device) for _ in range(len(x))]
                #     x_mask[i] = x_cur
                x[i] = x[i].repeat(m_cur.shape[0], 1)

            # print(sys.getsizeof(x_mask))
            # print(sys.getsizeof(x_mask[0]))

            # print('1', time.time()-t)
            ## 训练vae 重写一个模型
            # x_org = []
            # for _, c in enumerate(x):
            #     x_org.append(c.repeat(32,1))
            # t = time.time()
            # print('2', torch.cuda.memory_allocated(device))
            loss_vae, disc_loss = self.vae_model(x_mask, x)
            # print('3', torch.cuda.memory_allocated(device))
            # loss_vae, disc_loss = self.vae_model(x, x)
            mean = self.vae_model.mean
            self.pred = self.classifier(mean)
            # loss_clf = F.cross_entropy(self.pred,y,reduction='mean')
            loss = loss_vae
            # self.opt_vae.zero_grad()
            # loss.backward(retain_graph=True)
            self.opt_vae.zero_grad()
            # self.opt_cpl.zero_grad()
            # for fs in self.fs:
            #     fs.optimizer_cpl.zero_grad()
            loss.backward(retain_graph=True)
            self.opt_disc.zero_grad()
            # for fs in self.fs:
            #     fs.optimizer_cpl.step()
            # print('4', torch.cuda.memory_allocated(device))
            # print('5', torch.cuda.memory_allocated(device))
            disc_loss.backward()
            # print('6', torch.cuda.memory_allocated(device))
            # self.opt_cpl.step()
            self.opt_vae.step()
            self.opt_disc.step()
            # print('7', torch.cuda.memory_allocated(device))
            # print('2', time.time()-t)
            self.loss_clf = torch.tensor(0.)
            self.disc_loss = disc_loss
            self.loss_vae = loss

        else:
            # print('Iterative optimizating')

            ##########
            # pretrain_sel = self.e < self.num_batch_pretrain_sel
            # if self.e == self.num_batch_pretrain_sel:
            #     print('evoluting')
            #
            # self.e += 1
            #
            # if not pretrain_sel:
            #     # print('update M2')
            #     pass
            # m = []
            # # t = time.time()
            # for i in range(24):
            #     # print('training selector:{}'.format(i))
            #     fs = self.fs[i]
            #     m_cur = fs.get_m(pretrain=pretrain_sel)
            #     m.append(m_cur)
            #     x_cur, y_cur = fs.generate_train_data(x[i], m_cur.to(device), y)
            #     try:
            #         x_mask[i] = x_cur
            #     except:
            #         x_mask = [torch.zeros_like(x_cur, device=device) for _ in range(len(x))]
            #         x_mask[i] = x_cur
            #     x[i] = x[i].repeat(m_cur.shape[0], 1)
            #
            # # print(time.time() - t)
            # ## 训练vae 重写一个模型
            # # x_org = []
            # # for _, c in enumerate(x):
            # #     x_org.append(c.repeat(32, 1))
            # # loss_vae, disc_loss = self.vae_model(x_mask, x)
            # # mean = self.vae_model.mean
            # # self.pred = self.classifier(mean)
            # #
            # # # if pretrain_sel:
            # # #     loss_clf = F.cross_entropy(self.pred, y_cur, reduction='mean')
            # # # ## weighted_loss
            # # # else:
            # # #     loss_clf = F.cross_entropy(self.pred, y_cur, reduction='none')
            # # #     loss_clf = torch.mean(loss_clf.reshape(32,-1),dim=1)
            # # #     weight = torch.ones_like(loss_clf,device=device)
            # # #     weight[16] = 5.
            # # #     weight[17] = 10.
            # # #     loss_clf = torch.mean(loss_clf*weight)
            # # # loss = loss_clf + loss_vae
            # #
            # # loss_clf = F.cross_entropy(self.pred, y_cur, reduction='none')
            # # loss = torch.mean(loss_clf) + 0.01 * loss_vae
            # #
            # # self.opt_vae.zero_grad()
            # # self.opt_clf.zero_grad()
            # #
            # # loss.backward(retain_graph=True)
            # # self.opt_disc.zero_grad()
            # # disc_loss.backward()
            # # self.opt_vae.step()
            # # self.opt_disc.step()
            # # self.opt_clf.step()
            # #
            # # # self.loss_clf = loss_clf
            # # self.loss_clf = torch.mean(loss_clf)
            # # self.disc_loss = disc_loss
            # # self.loss_vae = loss_vae
            #
            # ## 预训练之后不计算重构损失
            # mean, _ = self.vae_model.encode(x_mask)
            # self.pred = self.classifier(mean)
            #
            # loss_clf = F.cross_entropy(self.pred, y_cur, reduction='none')
            #
            # loss = torch.mean(loss_clf)
            #
            # self.opt_clf.zero_grad()
            # self.opt_vae.zero_grad()
            # self.opt_cpl.zero_grad()
            # loss.backward()
            #
            # self.opt_clf.step()
            # self.opt_vae.step()
            # self.opt_cpl.step()
            # # self.loss_clf = loss_clf
            # self.loss_clf = torch.mean(loss_clf)
            # self.disc_loss = torch.tensor(0.)
            # self.loss_vae = torch.tensor(0.)
            #
            # for i in range(24):
            #     fs = self.fs[i]
            #     m_cur = m[i]
            #     fs.train_sel(m_cur, loss_clf, pretrain=pretrain_sel)

            ###########
            pretrain_sel = self.e < self.num_batch_pretrain_sel
            if self.e == self.num_batch_pretrain_sel:
                print('evoluting')

            self.e += 1
            if not pretrain_sel:
                # print('update M2')
                pass
            m = []
            # t = time.time()
            x_mask = []
            for i in range(24):
                fs = self.fs[i]
                x_mask.append(fs.complementary_feature.repeat(x[0].shape[0]*self.mask_size, 1))

            self.opt_clf.zero_grad()
            self.opt_vae.zero_grad()
            self.opt_cpl.zero_grad()

            for i in range(24):
                # print('training selector:{}'.format(i))
                fs = self.fs[i]
                m_cur = fs.get_m(pretrain=pretrain_sel)
                # m.append(m_cur)

                x_cur, y_cur = fs.generate_train_data(x[i], m_cur.to(device), y)
                tmp = x_mask[i]
                x_mask[i] = x_cur
                mean, _ = self.vae_model.encode(x_mask)
                x_mask[i] = tmp

                self.pred = self.classifier(mean)
                loss_clf = F.cross_entropy(self.pred, y_cur, reduction='none')

                loss = torch.mean(loss_clf)
                loss.backward()

                self.loss_clf = torch.mean(loss_clf)
                self.disc_loss = torch.tensor(0.)
                self.loss_vae = torch.tensor(0.)

                fs.train_sel(m_cur, loss_clf, pretrain=pretrain_sel)

            self.opt_clf.step()
            self.opt_vae.step()
            self.opt_cpl.step()


            ## 用一个小型分类器单独训练结果
            ## 这部分单独考虑一下

        self.print_m_opt = True


            ## 这里改一下
            # x_seg = [torch.zeros_like(c, device=device) for c in x_mask]
            # for i in range(24):
            #     fs = self.fs[i]
            #     # m_cur = fs.get_m(pretrain=pretrain_sel)
            #     # x_cur, y_cur = fs.generate_train_data(x[i], m_cur, y)
            #     # try:
            #     #     x_seg[i] = x_cur
            #     # except:
            #     #     x_seg = [torch.zeros_like(x_cur, device=device) for _ in range(len(x))]
            #     #     x_seg[i] = x_cur
            #     ## 这里是最不合理的地方
            #     ## 不是线性系统 不具有可加性
            #     x_cur,m_cur = x_mask[i],m[i]
            #     x_seg[i] = x_cur
            #     mean, _, _= self.vae_model.encode(x_seg)
            #     x_seg[i] = torch.zeros_like(x_cur, device=device)
            #     pred = self.classifier(mean)
            #     loss_sel = F.cross_entropy(pred, y_cur, reduction='none')
            #     fs.train_sel(m_cur, loss_sel, pretrain=pretrain_sel)


    def test(self,x,y):
        x_mask = []
        ## 获取 m_best 的步骤一次就行了
        for i in range(24):
            fs = self.fs[i]
            # fs.update_pop()
            # m_opt,_ = fs.get_importance()
            if self.print_m_opt:
                fs.update_m_opt()

            m_opt = fs.get_importance()
            # m_opt = m_opt.reshape(1,-1)
            if self.print_m_opt:
                chr = 'chr' + str(i + 1) if i < 22 else 'chrX'
                if i == 23:
                    chr = 'gene'
                print(chr, ':', torch.sum(m_opt.cpu()))
            x_cur, y_cur = fs.generate_train_data(x[i], m_opt.to(device), y)
            x_mask.append(x_cur)

        self.print_m_opt = False
        # print(x_mask[0].shape)
        # print(x_mask[-1].shape)
        mean, _ = self.vae_model.encode(x_mask)
        pred = self.classifier(mean)
        # print(pred.shape)
        return pred
'''
























