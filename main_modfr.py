import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from args import get_args_modfr

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
criterion = nn.MSELoss(reduction='none')

args = get_args_modfr()
root_path = args.data_path
mask_size_global = args.mask_size
random_mask_size = args.random_mask_size
E1 = args.E1
f = args.f
p = args.p


class MaskOptimizer:
    def __init__(self, feature_dim_list, unmasked_feature_size_list, mask_size, perturb_size, frac_randmask=0.5):
        self.feature_dim_list = feature_dim_list
        self.unmasked_size_list = unmasked_feature_size_list
        self.mask_size = mask_size          ## s
        self.perturb_size = perturb_size    ## sp
        self.frac_randmask = frac_randmask

        self.grid_dim = max(feature_dim_list)

    def get_random_mask_set(self, mask_size=0):
        if not mask_size:
            mask_size = self.mask_size
        num_omics = len(self.feature_dim_list)
        M = np.zeros(shape=(num_omics, mask_size, self.grid_dim))
        for i in range(num_omics):
            feature_dim, unmasked_size = self.feature_dim_list[i], self.unmasked_size_list[i]
            masks_zero = np.zeros(shape=(mask_size, feature_dim - unmasked_size))
            masks_one = np.ones(shape=(mask_size, unmasked_size))
            masks = np.concatenate([masks_zero, masks_one], axis=1)
            masks_permuted = np.apply_along_axis(np.random.permutation, 1, masks)

            masks_permuted = np.concatenate([masks_permuted, np.zeros(shape=(mask_size, self.grid_dim - feature_dim))], axis=1)
            M[i] += masks_permuted
        M = torch.tensor(M, dtype=torch.float32, device=device)
        return M

    def get_new_mask_set(self, last_best_mask):
        M = self.get_random_mask_set()
        idx = int(self.frac_randmask * self.mask_size)
        M[:, idx, :] = self.mask_opt
        M[:, idx+1, :] = last_best_mask
        M[:, idx+2:, :] = self.get_perturbed_masks(self.mask_size-(idx+2))
        return M


    def get_perturbed_masks(self, perturbed_mask_size):
        M_res = []
        def perturb_one_mask(mask):
            where_0 = np.nonzero(mask - 1)[0]
            where_1 = np.nonzero(mask)[0]
            if len(where_0):
                i0 = np.random.randint(0, len(where_0), 1)
                mask[where_0[i0]] = 1
            if len(where_1):
                i1 = np.random.randint(0, len(where_1), 1)
                mask[where_1[i1]] = 0
            return mask

        for i in range(len(self.feature_dim_list)):
            feature_dim = self.feature_dim_list[i]
            m_org = np.ones([perturbed_mask_size, feature_dim])*self.mask_opt.cpu().numpy()[i,:feature_dim]
            for _ in range(self.perturb_size):
                m_org = np.apply_along_axis(perturb_one_mask, 1, m_org)
            m_org = np.concatenate([m_org, np.zeros(shape=(perturbed_mask_size, self.grid_dim-feature_dim))], axis=1)
            m_org = torch.tensor(m_org, dtype=torch.float, device=device)
            m_org = m_org.unsqueeze(0)
            M_res.append(m_org)
        M_res = torch.cat(M_res, dim=0)
        return M_res


    def neg_gradient(self, selector_model, m):
        ## assert m.requirs_grad == True
        fs_m = selector_model(m)
        fs_m = torch.sum(fs_m)
        fs_m.backward()
        return - m.grad.squeeze(), fs_m


    def update_mask_opt(self, selector_model):

        M_epl = self.get_random_mask_set(random_mask_size)

        M_epl.requires_grad = True

        neg_grad, _ = self.neg_gradient(selector_model, M_epl)
        mask_opts = []
        self.importances = neg_grad.mean(1)
        neg_grad = neg_grad.cpu().mean(1).numpy()

        for i in range(len(self.feature_dim_list)):
            feature_dim, unmasked_size = self.feature_dim_list[i], self.unmasked_size_list[i]
            neg_grad_cur = neg_grad[i, :feature_dim]
            idx_unmasked_cur = list(np.argpartition(neg_grad_cur, -unmasked_size)[-unmasked_size:])
            m_opt_cur = torch.zeros([1, self.grid_dim], device=device)
            m_opt_cur[0, idx_unmasked_cur] = 1
            mask_opts.append(m_opt_cur)
        self.mask_opt = torch.cat(mask_opts, dim=0)

class SelectorNetwork(nn.Module):
    def __init__(self, feature_dim_list):
        super(SelectorNetwork, self).__init__()
        self.feature_dim_list = feature_dim_list

        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim_list[i], 64),
                                                   nn.ReLU()) for i in range(len(feature_dim_list))])
        self.out = nn.Sequential(nn.Linear(64*len(feature_dim_list), 128),
                                nn.ReLU(),
                                nn.Linear(128, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1))

    def forward(self, x):
        hidden = []
        for i, layer in enumerate(self.layers):
            hidden.append(layer(x[i, :, :self.feature_dim_list[i]]))
        hidden = torch.cat(hidden, dim=1)
        out = self.out(hidden)
        return out

class FeatureSelector:
    def __init__(self, feature_dim_list, unmasked_feature_size_list, mask_size, perturb_size=3, frac_randmask=0.5):
        self.mask_size = mask_size

        self.frac_randmask = frac_randmask
        self.feature_dim_list = feature_dim_list
        self.unmasked_size_list = unmasked_feature_size_list


        self.mask_optimizer = MaskOptimizer(feature_dim_list, unmasked_feature_size_list, mask_size, perturb_size,
                                            frac_randmask)


        self.selector = SelectorNetwork(feature_dim_list)
        self.selector = self.selector.to(device)

        self.e = 0

    def set_optimizer(self, lr_sel):
        self.optimizer_sel = Adam(self.selector.parameters(),
                                  lr=lr_sel)


    def generate_train_data(self, x, m, y):
        x_res = []
        if len(m.shape) == 2:
            m = m.unsqueeze(1)

        for i in range(len(self.feature_dim_list)):
            feature_dim = self.feature_dim_list[i]
            x_cur, m_cur = x[i], m[i, :, :feature_dim]
            x_size = x_cur.shape[0]
            m_size = m_cur.shape[0]
            x_cur = x_cur.repeat(m_size, 1)
            m_cur = m_cur.repeat(1, x_size).reshape(m_size * x_size, -1)
            x_cur = x_cur * m_cur
            x_res.append(x_cur)
        y = y.repeat(m_size)
        return x_res, y


    def get_weight_loss_sel(self):
        weight = torch.ones([self.mask_size], device=device)
        w0, w1 = 5, 10
        idx = int(self.mask_size * self.frac_randmask)
        weight[idx] = w0
        weight[idx + 1] = w1
        return weight


    def get_m(self, pretrain=True):
        if pretrain:
            M_cur = self.mask_optimizer.get_random_mask_set()
        else:
            update_m_opt_every_batches = 10
            if not self.e % update_m_opt_every_batches:
                self.mask_optimizer.update_mask_opt(self.selector)
            M_cur = self.mask_optimizer.get_new_mask_set(self.last_best_m_opt)
            self.e += 1

        return M_cur


    def train_sel(self, M_cur, loss_sel, pretrain):
        loss_sel = loss_sel.to(device)
        train_sel_x = M_cur
        train_sel_y = torch.mean(loss_sel.detach().reshape(M_cur.shape[1], -1), dim=1)

        self.last_best_m_opt = M_cur[:, torch.argsort(train_sel_y)[0], :]

        try:
            self.mask_optimizer.mask_opt = self.mask_optimizer.mask_opt
        except:
            self.mask_optimizer.mask_opt = self.last_best_m_opt

        try:
            self.mask_optimizer.importances = self.mask_optimizer.importances
        except:
            self.mask_optimizer.importances = torch.ones_like(self.mask_optimizer.mask_opt)

        fs_m = self.selector(train_sel_x)
        ## weighted mse_loss
        if pretrain:
            weight = torch.ones([self.mask_size], device=device)
        else:
            weight = self.get_weight_loss_sel()

        loss_sel = 0.5 * torch.mean(criterion(fs_m.squeeze(), train_sel_y) * weight)
        # print('loss_opt:',loss_opt.item(),'loss_sel:',loss_sel.item(),'Acc_pred',correct)
        self.optimizer_sel.zero_grad()
        loss_sel.backward()
        self.optimizer_sel.step()

    def update_m_opt(self):
        M_test = self.mask_optimizer.get_random_mask_set(mask_size=50)
        M_test.requires_grad = True
        neg_grad, _ = self.mask_optimizer.neg_gradient(self.selector, M_test)
        neg_grad = neg_grad.mean(0)
        idx_unmasked = list(np.argpartition(neg_grad.cpu().numpy(), -self.unmasked_size)[-self.unmasked_size:])
        # idx_unmasked = torch.where(neg_grad > 0)[0].tolist()
        m_opt = torch.zeros([self.feature_dim], device=device)
        m_opt[idx_unmasked] = 1
        m_opt = m_opt.unsqueeze(0)
        self.m_opt = m_opt

    def update_test(self):
        m_opt_last = self.mask_optimizer.mask_opt.detach().cpu()
        self.mask_optimizer.update_mask_opt(self.selector)
        m_opt_cur = self.mask_optimizer.mask_opt.detach().cpu()


    def get_importance(self, istest=False):
        return self.mask_optimizer.mask_opt, self.mask_optimizer.importances

class classifier(nn.Module):
    def __init__(self, latent_dim, label_dim=2):
        super(classifier, self).__init__()
        self.classifier_layer0 = nn.Sequential(nn.Linear(latent_dim, latent_dim//2),
                                               nn.ReLU(),
                                               )
        self.claasifier_layer1 = nn.Sequential(nn.Linear(latent_dim // 2, label_dim),
                                               )


    def forward(self, x):
        h0 = self.classifier_layer0(x)
        o = self.claasifier_layer1(h0)
        return o

class OperatorNetwork(nn.Module):
    def __init__(self, feature_dim_list, hidden_dim_list):
        super(OperatorNetwork, self).__init__()

        self.layers_0 = nn.ModuleList([nn.Sequential(nn.Linear(f, hidden_dim_list[0][0]),
                                                     nn.BatchNorm1d(hidden_dim_list[0][0]),
                                                     nn.ReLU(inplace=True)) for f in feature_dim_list[:-1]]
                                      +
                                      [nn.Sequential(nn.Linear(feature_dim_list[-1], hidden_dim_list[0][1]),
                                                     nn.BatchNorm1d(hidden_dim_list[0][1]),
                                                     nn.ReLU(inplace=True))])

        self.layers_2 = nn.Sequential(nn.Linear(sum(hidden_dim_list[0]), hidden_dim_list[2]),
                                      nn.BatchNorm1d(hidden_dim_list[2]),
                                      nn.ReLU(inplace=True))

        self.layers_mean = nn.Sequential(nn.Linear(hidden_dim_list[2], 128),
                                         nn.BatchNorm1d(128),
                                         )
        self.layers_logstd = nn.Sequential(nn.Linear(hidden_dim_list[2], 128),
                                         nn.BatchNorm1d(128),
                                         )


    def forward(self, x):
        hidden = []
        for i, layer in enumerate(self.layers_0):
            hidden.append(layer(x[i]))
        hidden = torch.cat(hidden, dim=1)
        out = self.layers_2(hidden)
        mean = self.layers_mean(out)
        logstd = self.layers_logstd(out)

        return mean,logstd,out

class DecoderNetwork(nn.Module):
    def __init__(self, feature_dim_list, hidden_dim_list):
        super(DecoderNetwork, self).__init__()
        n = len(feature_dim_list)

        self.hidden_dim_list = hidden_dim_list
        self.feature_dim_list = feature_dim_list

        self.layers_0 = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim_list[0][0], f),
                                                     nn.BatchNorm1d(f),
                                                     nn.Sigmoid(),
                                                     # nn.ReLU(inplace=True)
                                                     ) for f in feature_dim_list[:-1]]
                                      +
                                      [nn.Sequential(nn.Linear(hidden_dim_list[0][1], feature_dim_list[-1]),
                                                     nn.BatchNorm1d(feature_dim_list[-1]),
                                                     nn.Sigmoid(),
                                                     # nn.ReLU(inplace=True)
                                                     )])


        self.layers_2 = nn.Sequential(nn.Linear(hidden_dim_list[-1], sum(hidden_dim_list[0])),
                                      nn.BatchNorm1d(sum(hidden_dim_list[0])),
                                      nn.ReLU(inplace=True))

        self.layers_3 = nn.Sequential(nn.Linear(128, hidden_dim_list[-1]),
                                      nn.BatchNorm1d(hidden_dim_list[-1]),
                                      nn.ReLU(inplace=True))


    def forward(self, z):
        hidden = self.layers_3(z)
        hidden = self.layers_2(hidden)
        hidden = [hidden[:, :self.hidden_dim_list[0][0]], hidden[:, -self.hidden_dim_list[0][1]:]]
        for i, layer in enumerate(self.layers_0):
            hidden[i] = layer(hidden[i])

        return hidden

class DisNetwork(nn.Module):
    def __init__(self, hidden_dim_list):
        super(DisNetwork, self).__init__()
        self.dis_recon = nn.Sequential(nn.Linear(128, hidden_dim_list[-1]),
                                       nn.ReLU(inplace=True)
                                       )

        self.dis_layer0 = nn.Sequential(nn.Linear(hidden_dim_list[-1]*2, hidden_dim_list[-1]),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer1 = nn.Sequential(nn.Linear(hidden_dim_list[-1], hidden_dim_list[-1]//2),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer2 = nn.Sequential(nn.Linear(hidden_dim_list[-1]//2, hidden_dim_list[-1]//4),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer3 = nn.Linear(hidden_dim_list[-1]//4, 1)

    def forward(self, mean, z):
        recon = self.dis_recon(z)
        h = self.dis_layer0(torch.cat([mean, recon], dim=1))
        h = self.dis_layer1(h)
        h = self.dis_layer2(h)
        h = self.dis_layer3(h)
        return h

class MODFR(nn.Module):
    def __init__(self, feature_dim_list, hidden_dim_list, unmasked_size_list):
        super(MODFR, self).__init__()

        self.opt = OperatorNetwork(feature_dim_list, hidden_dim_list)
        self.dec = DecoderNetwork(feature_dim_list, hidden_dim_list)
        self.dis = DisNetwork(hidden_dim_list)

        mask_size = mask_size_global
        self.fs = FeatureSelector(feature_dim_list, unmasked_size_list, mask_size, perturb_size=p, frac_randmask=f)

        self.mask_size = mask_size
        self.classifier = classifier(128, label_dim=28) #label_dim=28
        self.e = 0

        self.num_batch_pretrain_sel = E1 ## 5

        self.print_m_opt = True

        n = len(feature_dim_list)
        self.n = n

    def init_opt(self):
        self.fs.set_optimizer(0.001)

        self.opt_clf = Adam(self.classifier.parameters(), lr=0.001)
        self.opt_vae = Adam(self.opt.parameters(),
                            lr = 0.001)

        self.opt_dec = Adam(self.dec.parameters(),
                            lr = 0.001)
        self.opt_dis = Adam(self.dis.parameters(),
                            lr = 0.001)

    def sample_prior(self, z):
        p = torch.randn_like(z, device=device)
        return p

    def sample_posterior(self, mean, logstd=1):
        gaussian_noise = torch.randn_like(mean, device=device)
        p = mean + gaussian_noise * torch.exp(logstd)
        return p

    def discriminator(self, mean, z):
        return self.dis(mean, z)

    def recon_loss(self, x_list, recon_x_list):
        loss = 0
        for x, recon_x in zip(x_list[:-1], recon_x_list[:-1]):
            cur_loss = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).mean() ## 'sum'
            # cur_loss = F.binary_cross_entropy(recon_x, x, reduction='none').mean()
            loss += cur_loss
        # loss = loss / 23
        loss = loss / (self.n - 1)
        loss = loss + F.binary_cross_entropy(recon_x_list[-1], x_list[-1], reduction='none').sum(1).mean()
        # loss = loss + F.binary_cross_entropy(recon_x_list[-1], x_list[-1], reduction='none').mean()
        return loss

    def train_avb(self, x):
        mean, logstd, hidden_integrated = self.opt(x)
        z_q = self.sample_posterior(mean, logstd)
        z_p = self.sample_prior(z_q)
        log_posterior = self.discriminator(hidden_integrated, z_q)
        log_prior_dis = self.discriminator(hidden_integrated.detach(), z_p)
        log_posterior_dis = self.discriminator(hidden_integrated.detach(), z_q.detach())

        loss_disc = torch.sum(F.binary_cross_entropy_with_logits(log_posterior_dis, torch.ones_like(log_posterior_dis),
                                                                 reduction='none').sum(1).mean()
                              + F.binary_cross_entropy_with_logits(log_prior_dis, torch.zeros_like(log_prior_dis),
                                                                   reduction='none').sum(1).mean())

        kl = log_posterior.mean()
        # print(z_q.shape)
        recon_list = self.dec(z_q)

        loss_recon = self.recon_loss(x, recon_list)
        # loss_recon = self.recon_loss(x_repeat, recon_list)

        loss = loss_recon + kl

        self.opt_vae.zero_grad()
        self.opt_dec.zero_grad()

        loss.backward(retain_graph=True)
        self.opt_dis.zero_grad()
        loss_disc.backward()

        self.opt_vae.step()
        self.opt_dec.step()
        self.opt_dis.step()

        print('loss_recon:', loss_recon.item(), 'loss_kl:', kl.item(), 'loss_dis:', loss_disc.item())


    def train_model(self, x, y):

        pretrain_sel = self.e < self.num_batch_pretrain_sel
        if self.e == self.num_batch_pretrain_sel:
            print('evoluting')
        self.e += 1

        m = self.fs.get_m(pretrain=pretrain_sel)
        # x_mask, y = self.fs.generate_train_data(x, m.to(device), y)
        x_mask, y = self.fs.generate_train_data(x, m, y)

        # mean, _ = self.vae_model.encode(x_mask)
        mean,_,_ = self.opt(x_mask)

        self.pred = self.classifier(mean)

        loss_clf = F.cross_entropy(self.pred, y, reduction='none')
        # loss = torch.mean(loss_clf)
        loss_train_sel = torch.ones_like(loss_clf, device=device) * loss_clf

        loss = loss_clf.reshape(m[0].shape[0], -1)
        idx = int(m[0].shape[0]*0.5)

        loss_opt = loss[idx, :]
        self.loss_opt = torch.mean(loss_opt).item()

        loss = torch.mean(loss)
        self.loss_train = loss.item()

        self.opt_clf.zero_grad()
        self.opt_vae.zero_grad()

        loss.backward()
        self.opt_clf.step()
        self.opt_vae.step()

        self.loss_clf = loss
        self.disc_loss = torch.tensor(0.)
        self.loss_vae = torch.tensor(0.)

        self.fs.train_sel(m, loss_train_sel, pretrain=pretrain_sel)

        self.print_m_opt = True


    def test(self, x, y):
        m_opt, _ = self.fs.get_importance()

        x_mask, _ = self.fs.generate_train_data(x, m_opt, y)
        self.print_m_opt = False
        mean,_,_ = self.opt(x_mask)
        pred = self.classifier(mean)
        self.loss_opt_val = F.cross_entropy(pred,y).item()

        return pred


def dataset_split(i=1):
    filter_samples = pd.read_csv(root_path + 'sub_dataset/subsamples_idx_nonormal.tsv', sep='\t', index_col=0)
    filter_list = filter_samples.to_numpy().squeeze()
    labels = pd.read_csv(root_path + 'sub_dataset/subsamples_labels_nonormal.tsv', sep='\t', index_col=0)

    print('Loading gene expression data...')
    gene_expression = pd.read_csv(root_path + 'sub_dataset/gene_expression_co10_filtered.tsv', sep='\t',
                                  index_col=0)
    print(gene_expression.shape)

    print('Loading DNA methylation data...')
    dna_methylation = pd.read_csv(root_path + 'sub_dataset/dna_methylation_co10_filtered.tsv', sep='\t', index_col=0)
    dna_methylation = dna_methylation.filter(items=filter_list, axis=1)
    print(dna_methylation.shape)

    feature_dim_list = []
    feature_dim_list.append(dna_methylation.shape[0])
    feature_dim_list.append(gene_expression.shape[0])

    ## 5 fold 验证
    print('loading fold', str(i))
    drop_samples = pd.read_csv(root_path + 'sub_dataset/subsamples_idx_' + str(i) + 'fold.tsv', sep='\t', index_col=0)
    drop_samples = drop_samples.to_numpy().squeeze()

    gene_expression_train = gene_expression.drop(labels=drop_samples, axis=1)
    dna_methylation_train = dna_methylation.drop(labels=drop_samples, axis=1)
    train_label = labels.drop(labels=drop_samples, axis=0)

    train_label = train_label.to_numpy().squeeze()
    print(np.unique(train_label))

    gene_expression_test = gene_expression.filter(items=drop_samples, axis=1)
    dna_methylation_test = dna_methylation.filter(items=drop_samples, axis=1)
    test_label = labels.filter(items=drop_samples, axis=0)
    test_label = test_label.to_numpy().squeeze()


    gene_expression_train = gene_expression_train.to_numpy().transpose()
    gene_expression_train = torch.FloatTensor(gene_expression_train)


    dna_methylation_train = dna_methylation_train.to_numpy().transpose()
    dna_methylation_train = torch.FloatTensor(dna_methylation_train)


    print(gene_expression_train.shape)

    gene_expression_test = gene_expression_test.to_numpy().transpose()
    gene_expression_test = torch.FloatTensor(gene_expression_test)

    dna_methylation_test = dna_methylation_test.to_numpy().transpose()
    dna_methylation_test = torch.FloatTensor(dna_methylation_test)

    train_label = torch.LongTensor(train_label)
    test_label = torch.LongTensor(test_label)

    return gene_expression_train, dna_methylation_train, train_label, gene_expression_test, dna_methylation_test, test_label, feature_dim_list


class BatchIndex:
    def __init__(self, size, batch_size, shuffle=False, drop_last=True):
        idx = np.arange(size)
        if shuffle:
            np.random.shuffle(idx)

        self.index_list = [idx[x:x + batch_size] for x in range(0, size, batch_size)]

        if not drop_last:
            self.index_list.append(idx[-size % batch_size:])

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index_list):
            raise StopIteration
        return self.index_list[self.pos]

    def __iter__(self):
        self.pos = -1
        return self

    def __len__(self):
        return len(self.index_list)

def main(i=1):
    gene_expression_train, dna_methylation_train, train_label, gene_expression_test, dna_methylation_test, test_label, features_dim_list = dataset_split(i=i)
    print(features_dim_list)

    samples_train_size = gene_expression_train.shape[0]
    gene_expression_train = gene_expression_train.to(device)
    dna_methylation_train = dna_methylation_train.to(device)
    train_label = train_label.to(device)

    gene_expression_test = gene_expression_test.to(device)
    dna_methylation_test = dna_methylation_test.to(device)
    test_label = test_label.to(device)

    hidden_dim_list = [(1024, 1024), (512, 512), 256]
    unmasked_size_list = [args.FS] * len(features_dim_list)

    model = MODFR(features_dim_list, hidden_dim_list, unmasked_size_list)
    model.init_opt()
    model = model.to(device)

    train_loss = []
    train_loss_opt = []
    val_loss_opt = []
    operator_acc_train = []
    operator_acc_val = []

    if args.use_pr:
        print('loading parameters')
        model.opt.load_state_dict(torch.load(args.pretrained_encoder))

    epochs = args.epochs
    for epoch in range(1, epochs + 1):
        for batch, idx in enumerate(BatchIndex(samples_train_size, args.batch_size, shuffle=True, drop_last=True)):
            features = [dna_methylation_train[idx, :], gene_expression_train[idx, :]]
            label = train_label[idx]
            batch_size = label.shape[0]

            model.train_model(features, label)

            train_loss.append(model.loss_train)
            train_loss_opt.append(model.loss_opt)

            with torch.no_grad():
                pred = model.pred
                pred = F.softmax(pred, dim=1)
                _, pred = torch.max(pred, dim=1)

                mask_size = mask_size_global

                pred = pred.reshape(mask_size, -1)
                idx = mask_size // 2
                pred_random = pred[:idx]
                pred_last_best = pred[idx + 1]
                pred_m_opt = pred[idx]
                pred_perturb = pred[idx + 2:]

                correct_rand = (pred_random.reshape(-1) == label.repeat(idx)).sum().item()
                correct_last_best = (pred_last_best == label).sum().item()
                correct_m_opt = (pred_m_opt == label).sum().item()
                correct_perturb = (pred_perturb.reshape(-1) == label.repeat(mask_size - idx - 2)).sum().item()

                num_pred_correct_rand = correct_rand / idx
                num_pred_correct_last = correct_last_best
                num_pred_correct_mopt = correct_m_opt
                num_pred_correct_perturb = correct_perturb / (mask_size - idx - 2)

                acc_rand = num_pred_correct_rand / batch_size * 100
                acc_last = num_pred_correct_last / batch_size * 100
                acc_m_opt = num_pred_correct_mopt / batch_size * 100
                acc_perturb = num_pred_correct_perturb / batch_size * 100

            print("Epoch:{:3d},Batch:{:3d},acc_rand:{:.3f}%,acc_last:{:.3f}%,acc_m_opt:{:.3f}%,acc_perturb:{:.3f}%".format(epoch,
                                                                                                                           batch,
                                                                                                                           acc_rand,
                                                                                                                           acc_last,
                                                                                                                           acc_m_opt,
                                                                                                                           acc_perturb))

            features = [dna_methylation_test, gene_expression_test]
            label = test_label
            batch_size = label.shape[0]

            pred = model.test(features, label)
            val_loss_opt.append(model.loss_opt_val)

            pred = F.softmax(pred, dim=1)
            _, pred = torch.max(pred, dim=1)
            correct = (pred == label).sum().item()
            num_pred_correct = correct
            acc_val = num_pred_correct / batch_size * 100

            operator_acc_train.append(acc_m_opt)
            operator_acc_val.append(acc_val)

        # if epoch >= 100 and not epoch % 100:
        #     _, importances = model.fs.get_importance()
        #     importances = pd.DataFrame(importances.cpu().detach().numpy().transpose())
        #     # importances.to_csv('/ibgpu01/lmh/multi-omics/database/3.28/MODFR_3_' + str(epoch) + '_fold' + str(i) + '.tsv', sep='\t', index=False)
        #     save_file = args.save_path + 'importances_fold' + str(i) +'_'+ str(epoch) + '.tsv'
        #     importances.to_csv(save_file, sep='\t', index=False)

    _, importances = model.fs.get_importance()
    importances = pd.DataFrame(importances.cpu().detach().numpy().transpose())
    save_file = args.save_path + 'importances_fold'+ str(i) + '.tsv'
    importances.to_csv(save_file, sep='\t', index=False)

    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')
    plt.ylabel('accuracy')

    plt.plot(np.arange(len(operator_acc_train)), np.array(operator_acc_train), color='red', linestyle="solid", label="train accuracy")
    plt.plot(np.arange(len(operator_acc_val)), np.array(operator_acc_val), color='blue', linestyle="solid", label="val accuracy")
    plt.legend()
    plt.title('Accuracy curve')
    plt.show()

    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(np.arange(len(train_loss)), np.array(train_loss), color='red', linestyle="solid", label="train loss")
    plt.plot(np.arange(len(train_loss_opt)), np.array(train_loss_opt), color='green', linestyle="solid", label="train loss(m_opt)")
    plt.plot(np.arange(len(operator_acc_val)), np.array(val_loss_opt), color='blue', linestyle="solid", label="val loss")
    plt.legend()
    plt.title('loss curve')
    plt.show()

if __name__ == '__main__':
    for i in range(1, 5 + 1):
        main(i=i)