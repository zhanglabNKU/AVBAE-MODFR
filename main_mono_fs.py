import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mask_size_global = 128

class Subdataset(Dataset):
    def __init__(self, gene_expression, label):
        gene_expression = gene_expression.to_numpy().transpose()
        gene_expression = torch.FloatTensor(gene_expression)
        # gene_expression = gene_expression.to(device)
        labels = torch.LongTensor(label)
        # labels = labels.to(device)

        self.gene_expression = gene_expression
        self.labels = labels
    def __len__(self):
        return self.gene_expression.shape[0]

    def __getitem__(self, item):
        labels = self.labels[item]
        expression = self.gene_expression[item,:]
        return expression, labels


def dataset_split(i=1, dataset='exp'):
    filter_samples = pd.read_csv('./data/sub_dataset/subsamples_idx_nonormal.tsv', sep='\t', index_col=0)
    filter_list = filter_samples.to_numpy().squeeze()
    labels = pd.read_csv('./data/sub_dataset/subsamples_labels_nonormal.tsv', sep='\t', index_col=0)

    if dataset == 'exp':
        print('Loading gene expression data...')
        gene_expression = pd.read_csv('./data/sub_dataset/gene_expression_co10_filtered.tsv', sep='\t',
                                      index_col=0)
        print(gene_expression.shape)
        data = gene_expression

    elif dataset == 'met':
        print('Loading DNA methylation data...')
        dna_methylation = pd.read_csv('./data/sub_dataset/dna_methylation_co10_filtered.tsv', sep='\t', index_col=0)
        dna_methylation = dna_methylation.filter(items=filter_list, axis=1)
        print(dna_methylation.shape)
        data = dna_methylation
    else:
        raise ValueError('Dataset{} not exit'.format(dataset))


    feature_dim = data.shape[0]



    ## 5 fold 验证
    print('loading fold', str(i))
    drop_samples = pd.read_csv('./data/sub_dataset/subsamples_idx_' + str(i) + 'fold.tsv', sep='\t', index_col=0)
    drop_samples = drop_samples.to_numpy().squeeze()

    data_train = data.drop(labels=drop_samples, axis=1)
    train_label = labels.drop(labels=drop_samples, axis=0)

    train_label = train_label.to_numpy().squeeze()

    data_test = data.filter(items=drop_samples, axis=1)
    test_label = labels.filter(items=drop_samples, axis=0)
    test_label = test_label.to_numpy().squeeze()

    data_train = data_train.to_numpy().transpose()
    data_train = torch.FloatTensor(data_train)

    data_test = data_test.to_numpy().transpose()
    data_test = torch.FloatTensor(data_test)

    train_label = torch.LongTensor(train_label)
    test_label = torch.LongTensor(test_label)
    return data_train, train_label, data_test, test_label, feature_dim


class MaskOptimizer():
    def __init__(self, feature_dim, unmasked_feature_size, mask_size, perturb_size, frac_randmask=0.5):
        self.feature_dim = feature_dim
        self.unmasked_size = unmasked_feature_size
        self.mask_size = mask_size          ## s
        self.perturb_size = perturb_size    ## sp
        self.frac_randmask = frac_randmask

    def get_random_mask_set(self, mask_size=0):
        if not mask_size:
            mask_size = self.mask_size
        ## for pretrain
        masks_zero = np.zeros(shape=(mask_size, self.feature_dim - self.unmasked_size))
        masks_one = np.ones(shape=(mask_size, self.unmasked_size))
        masks = np.concatenate([masks_zero, masks_one], axis=1)
        masks_permuted = np.apply_along_axis(np.random.permutation, 1, masks)
        M = torch.tensor(masks_permuted, dtype=torch.float32)
        M = M.to(device)
        return M


    def get_new_mask_set(self, last_best_mask):
        M = self.get_random_mask_set()
        idx = int(self.frac_randmask * self.mask_size)
        M[idx] = self.mask_opt
        M[idx+1] = last_best_mask
        M[idx+2:] = self.get_perturbed_masks(self.mask_size-(idx+2))
        return M

    def get_perturbed_masks(self, perturbed_mask_size):

        M = np.ones([perturbed_mask_size, self.mask_opt.shape[1]]) * self.mask_opt.cpu().numpy()
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

        for _ in range(self.perturb_size): ##
            M = np.apply_along_axis(perturb_one_mask, 1, M)

        return torch.tensor(M, dtype=torch.float, device=device)


    def neg_gradient(self, selector_model, m):
        ## assert m.requirs_grad == True
        fs_m = selector_model(m)
        fs_m = torch.sum(fs_m)

        fs_m.backward()

        return -m.grad.squeeze(), fs_m

    def update_mask_opt(self, selector_model, max_steps=5):

        t = time.time()
        M_size = 50
        M_epl = self.get_random_mask_set(M_size)
        M_epl.requires_grad = True
        neg_grad, _ = self.neg_gradient(selector_model, M_epl)
        neg_grad = neg_grad.mean(0)
        # idx_unmasked = torch.where(neg_grad > 0)[0].tolist()
        idx_unmasked = list(np.argpartition(neg_grad.cpu().numpy(), -self.unmasked_size)[-self.unmasked_size:])
        m_opt = torch.zeros([self.feature_dim], device=device)
        m_opt[idx_unmasked] = 1
        self.mask_opt = m_opt.unsqueeze(0).detach()

        print('Time for one-time updating mask_opt:(M_size = {})'.format(M_size), time.time() - t)


criterion = nn.MSELoss(reduction='none')

class OperatorNetwork(nn.Module):
    def __init__(self, feature_dim, label_dim=28):
        super(OperatorNetwork, self).__init__()
        ## Dense
        self.layer = nn.Sequential(nn.Linear(feature_dim, 4096),
                                   nn.BatchNorm1d(4096),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(4096, 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024, 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(),

                                   nn.Linear(512, label_dim),
                                   )

    def forward(self, x):
        return self.layer(x)


class SelectorNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(SelectorNetwork, self).__init__()
        self.layer = nn.Sequential(nn.Linear(feature_dim, 512),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(512, 256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 128),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, 1))
    def forward(self,m):

        return self.layer(m)


class OptNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(OptNetwork, self).__init__()
        self.layer = nn.Sequential(nn.Linear(feature_dim, 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),

                                   )
        self.layer_mean = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        )
        self.layer_logstd = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        )

    def forward(self, x):
        out = self.layer(x)
        mean = self.layer_mean(out)
        logstd = self.layer_logstd(out)
        return mean, logstd, out


class DecNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(DecNetwork, self).__init__()
        self.layer = nn.Sequential(nn.Linear(128, 256),
                                   nn.BatchNorm1d(256),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(256, 1024),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024, feature_dim),
                                   nn.BatchNorm1d(feature_dim),
                                   nn.Sigmoid(),
                                   )

    def forward(self,z):
        return self.layer(z)

class DisNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(DisNetwork, self).__init__()
        self.dis_recon = nn.Sequential(nn.Linear(128, hidden_dim),
                                       nn.ReLU(inplace=True)
                                       )

        self.dis_layer0 = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer2 = nn.Sequential(nn.Linear(hidden_dim//2, hidden_dim//4),
                                        nn.ReLU(inplace=True)
                                        )

        self.dis_layer3 = nn.Linear(hidden_dim//4, 1)

    def forward(self, mean, z):
        recon = self.dis_recon(z)
        h = self.dis_layer0(torch.cat([mean, recon], dim=1))
        h = self.dis_layer1(h)
        h = self.dis_layer2(h)
        h = self.dis_layer3(h)
        return h


class ClfNetwork(nn.Module):
    def __init__(self,label_dim):
        super(ClfNetwork, self).__init__()
        latent_dim = 128
        self.classifier_layer0 = nn.Sequential(nn.Linear(latent_dim, latent_dim//2),
                                               # nn.BatchNorm1d(latent_dim//2),
                                               # nn.Linear(latent_dim, latent_dim),
                                               # nn.BatchNorm1d(latent_dim),
                                               nn.ReLU(inplace=True),
                                               # nn.Dropout(p=0.5)
                                               )
        self.claasifier_layer1 = nn.Sequential(nn.Linear(latent_dim // 2, label_dim),
                                               # nn.BatchNorm1d(label_dim),
                                               )

    def forward(self, x):
        h0 = self.classifier_layer0(x)
        o = self.claasifier_layer1(h0)
        return o


class FeatureSelector():
    def __init__(self, feature_dim, unmasked_feature_size, mask_size, label_dim=28, frac_randmask=0.5):
        self.feature_dim = feature_dim
        self.unmasked_size = unmasked_feature_size

        self.perturb_size = 5
        self.mask_size = mask_size

        self.frac_randmask = frac_randmask

        self.mask_optimizer = MaskOptimizer(feature_dim, unmasked_feature_size, mask_size, self.perturb_size, frac_randmask)


        hidden_dim = 256
        self.opt = OptNetwork(feature_dim)
        self.dec = DecNetwork(feature_dim)
        self.dis = DisNetwork(hidden_dim)
        self.clf = ClfNetwork(label_dim)

        self.selector = SelectorNetwork(feature_dim)
        self.selector = self.selector.to(device)

        self.E1 = 200 ## DNA methylation
        self.E1 = 150 ## Gene expression
        self.e = 0


    def set_optimizer(self, lr_opt, lr_sel):

        self.optimizer_sel = Adam(self.selector.parameters(),
                                  lr=lr_sel)


        self.opt_dec = Adam(self.dec.parameters(),
                            lr=0.001)
        self.opt_dis = Adam(self.dis.parameters(),
                            lr=0.001)
        self.opt_clf = Adam(self.clf.parameters(),
                            lr=0.001)
        self.opt_enc = Adam(self.opt.parameters(),
                            lr=0.001)


    def generate_train_data(self, x, m, y):
        x_size = x.shape[0]
        m_size = m.shape[0]
        x = x.repeat(m_size, 1)
        m = m.repeat(1, x_size).reshape(m_size*x_size, -1)
        x = x*m

        y = y.repeat(m_size)
        return x, y


    def get_weight_loss_sel(self):
        weight = torch.ones([self.mask_size], device=device)
        if self.e < self.E1:
            return weight
        # w0, w1 = 5, 10
        w0, w1 = 10, 5
        idx = int(self.mask_size * self.frac_randmask)
        weight[idx] = w0
        weight[idx+1] = w1

        return weight

    ## 测试
    def test(self, x, m, y):

        # self.operator.eval()
        # self.selector.eval()
        self.opt.eval()
        self.clf.eval()

        test_x, test_y = self.generate_train_data(x, m, y)

        mean, _, _ = self.opt(test_x)
        pred = self.clf(mean)


        loss = F.cross_entropy(pred, test_y, reduction='mean')
        self.loss_opt_val = loss

        self.opt.train()
        self.clf.train()

        return pred


    def train_opt(self, x, y):
        output_opt = self.operator(x)
        loss_opt_org = F.cross_entropy(output_opt, y, reduction='mean')

        self.optimizer_opt.zero_grad()
        loss_opt_org.backward()
        self.optimizer_opt.step()

    def sample_prior(self, z):
        p = torch.randn_like(z, device=device)
        return p

    def sample_posterior(self, mean, logstd=1):
        gaussian_noise = torch.randn_like(mean, device=device)
        p = mean + gaussian_noise * torch.exp(logstd)
        return p

    def train_avb(self,x):
        mean, logstd, hidden_integrated = self.opt(x)
        z_q = self.sample_posterior(mean, logstd)
        z_p = self.sample_prior(z_q)

        log_posterior = self.dis(hidden_integrated, z_q)
        log_prior_dis = self.dis(hidden_integrated.detach(), z_p)
        log_posterior_dis = self.dis(hidden_integrated.detach(), z_q.detach())
        loss_disc = torch.sum(F.binary_cross_entropy_with_logits(log_posterior_dis, torch.ones_like(log_posterior_dis),
                                                                 reduction='none').sum(1).mean()
                              + F.binary_cross_entropy_with_logits(log_prior_dis, torch.zeros_like(log_prior_dis),
                                                                   reduction='none').sum(1).mean())

        kl = log_posterior.mean()
        recon_x = self.dec(z_q)

        loss_recon = F.binary_cross_entropy(recon_x, x, reduction='none').sum(1).mean()

        loss = loss_recon + kl


        self.opt_enc.zero_grad()
        self.opt_dec.zero_grad()

        loss.backward(retain_graph=True)
        self.opt_dis.zero_grad()
        loss_disc.backward()
        self.opt_enc.step()
        self.opt_dec.step()
        self.opt_dis.step()

        print('loss_recon:', loss_recon.item(), 'loss_kl:', kl.item(), 'loss_dis:', loss_disc.item())


    def train_sel(self, x, y):
        weight = self.get_weight_loss_sel()
        if self.e < self.E1:
            self.e += 1
            M_cur = self.mask_optimizer.get_random_mask_set()
        else:
            if self.e == self.E1:
                print('_________________________________________')
            self.e += 1
            update_m_opt_every_batches = 10
            if not self.e % update_m_opt_every_batches:

                self.mask_optimizer.update_mask_opt(self.selector)

            M_cur = self.mask_optimizer.get_new_mask_set(self.last_best_m_opt)

        train_opt_x, train_opt_y = self.generate_train_data(x, M_cur, y)

        output_opt = self.operator(train_opt_x)

        loss_opt_org = F.cross_entropy(output_opt, train_opt_y, reduction='none')
        m_size = M_cur.shape[0]
        idx = int(m_size * 0.5)

        loss_sel = torch.ones_like(loss_opt_org, device=device) * loss_opt_org

        _, pred = torch.max(output_opt, dim=1)

        pred = pred.reshape(m_size, -1)
        pred_random = pred[:idx]
        pred_last_best = pred[idx + 1]
        pred_m_opt = pred[idx]
        pred_perturb = pred[idx + 2:]

        self.correct_rand = (pred_random.reshape(-1) == y.repeat(idx)).sum().item() / idx
        self.correct_last_best = (pred_last_best == y).sum().item()
        self.correct_m_opt = (pred_m_opt == y).sum().item()
        self.correct_perturb = (pred_perturb.reshape(-1) == y.repeat(self.mask_size - idx - 2)).sum().item() / (
                    self.mask_size - idx - 2)

        train_sel_x = M_cur
        train_sel_y = torch.mean(loss_sel.detach().reshape(M_cur.shape[0], -1), dim=1)

        self.last_best_m_opt = M_cur[torch.argsort(train_sel_y)[0]]
        fs_m = self.selector(train_sel_x)

        loss_sel = 0.5 * torch.mean(criterion(fs_m.squeeze(), train_sel_y) * weight)

        self.optimizer_sel.zero_grad()
        loss_sel.backward()
        self.optimizer_sel.step()


    def train(self, x, y):
        weight = self.get_weight_loss_sel()
        if self.e < self.E1:
            ## 1234567
            self.e += 1
            M_cur = self.mask_optimizer.get_random_mask_set()
        else:
            ## 8-23
            if self.e == self.E1:
                print('_________________________________________')
            self.e += 1
            update_m_opt_every_batches = 10
            if not self.e % update_m_opt_every_batches:
                self.mask_optimizer.update_mask_opt(self.selector)

            M_cur = self.mask_optimizer.get_new_mask_set(self.last_best_m_opt)

        train_opt_x, train_opt_y = self.generate_train_data(x, M_cur, y)
        mean,_,_ = self.opt(train_opt_x)
        output_opt = self.clf(mean)

        loss_opt_org = F.cross_entropy(output_opt, train_opt_y, reduction='none')
        m_size = M_cur.shape[0]
        loss_opt = loss_opt_org.reshape(m_size, -1)
        idx = int(m_size * 0.5)

        loss_opt[:idx] = loss_opt[:idx] * 1
        loss_opt[idx] = loss_opt[idx] * 1
        loss_opt[idx+1] = loss_opt[idx+1] * 1


        self.loss_opt = torch.mean(loss_opt[idx])
        loss_opt = torch.mean(loss_opt)
        self.loss_train = loss_opt

        loss_sel = torch.ones_like(loss_opt_org, device=device) * loss_opt_org

        _, pred = torch.max(output_opt, dim=1)

        pred = pred.reshape(m_size, -1)
        pred_random = pred[:idx]
        pred_last_best = pred[idx + 1]
        pred_m_opt = pred[idx]
        pred_perturb = pred[idx + 2:]

        self.correct_rand = (pred_random.reshape(-1) == y.repeat(idx)).sum().item() / idx
        self.correct_last_best = (pred_last_best == y).sum().item()
        self.correct_m_opt = (pred_m_opt == y).sum().item()
        self.correct_perturb = (pred_perturb.reshape(-1) == y.repeat(self.mask_size - idx - 2)).sum().item() / (self.mask_size-idx-2)

        train_sel_x = M_cur
        train_sel_y = torch.mean(loss_sel.detach().reshape(M_cur.shape[0], -1), dim=1)

        self.last_best_m_opt = M_cur[torch.argsort(train_sel_y)[0]]

        fs_m = self.selector(train_sel_x)

        loss_sel = 0.5 * torch.mean(criterion(fs_m.squeeze(), train_sel_y)*weight)

        self.loss_sel = loss_sel


        self.opt_enc.zero_grad()
        self.opt_clf.zero_grad()
        loss_opt.backward()
        self.opt_enc.step()
        self.opt_clf.step()

        self.optimizer_sel.zero_grad()
        loss_sel.backward()
        self.optimizer_sel.step()

    def get_importance(self):
        M_test = self.mask_optimizer.get_random_mask_set(mask_size=500)
        M_test.requires_grad = True
        neg_grad, _ = self.mask_optimizer.neg_gradient(self.selector, M_test)
        neg_grad = neg_grad.mean(0)

        try:
            m_opt = self.mask_optimizer.mask_opt
        except:
            idx_unmasked = list(np.argpartition(neg_grad.cpu().numpy(), -self.unmasked_size)[-self.unmasked_size:])
            m_opt = torch.zeros([self.feature_dim], device=device)
            m_opt[idx_unmasked] = 1
            m_opt = m_opt.unsqueeze(0)

        return m_opt, neg_grad

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


def train_fs_ite(i=1, dataset='exp'):
    train_feature, train_label, test_feature, test_label, feature_dim = dataset_split(i=i, dataset=dataset)
    train_feature = train_feature.to(device)
    train_label = train_label.to(device)
    test_feature = test_feature.to(device)
    test_label = test_label.to(device)
    fs = FeatureSelector(feature_dim, 50, mask_size_global, label_dim=10)

    fs.set_optimizer(0.001, 0.001)

    fs.opt = fs.opt.to(device)
    fs.dec = fs.dec.to(device)
    fs.dis = fs.dis.to(device)
    fs.clf = fs.clf.to(device)

    train_loss = []
    train_loss_opt = []
    val_loss_opt = []
    operator_acc_train = []
    operator_acc_val = []
    cnt = 0
    epochs = 50
    acc_best = 0
    patience = 100
    isbreak = False

    for epoch in range(1, epochs+1, 1):
        for batch, idx in enumerate(BatchIndex(train_feature.shape[0], 128, shuffle=True, drop_last=True)):
            x = train_feature[idx, :]
            y = train_label[idx]

            fs.train(x, y)
            batch_size = y.shape[0]

            num_pred_correct_rand = fs.correct_rand
            num_pred_correct_last = fs.correct_last_best
            num_pred_correct_mopt = fs.correct_m_opt
            num_pred_correct_perturb = fs.correct_perturb

            acc_rand = num_pred_correct_rand / batch_size * 100
            acc_last = num_pred_correct_last / batch_size * 100
            acc_m_opt = num_pred_correct_mopt / batch_size * 100
            acc_perturb = num_pred_correct_perturb / batch_size * 100

            train_loss.append(fs.loss_train.item())
            train_loss_opt.append(fs.loss_opt.item())

            x, y = test_feature, test_label
            m_best, _ = fs.get_importance()

            pred = fs.test(x, m_best, y)
            pred = F.softmax(pred, dim=1)
            _, pred = torch.max(pred, dim=1)
            correct = (pred == y).sum().item()
            num_pred_correct = correct
            batch_size_val = y.shape[0]
            acc_val = num_pred_correct / batch_size_val * 100

            val_loss_opt.append(fs.loss_opt_val.item())
            operator_acc_train.append(acc_m_opt)
            operator_acc_val.append(acc_val)

            print("Epoch:{:3d},Batch:{:3d},acc_rand:{:.3f}%,acc_last:{:.3f}%,acc_m_opt:{:.3f}%,acc_perturb:{:.3f}%".format(
                    epoch,
                    batch,
                    acc_rand,
                    acc_last,
                    acc_m_opt,
                    acc_perturb))


        if epoch >= 30 and not epoch % 10:
            _, importances = fs.get_importance()
            importances = pd.DataFrame(importances.cpu().detach().numpy().transpose())

            # importances.to_csv(save_path, sep='\t', index=False)


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

    # operator_acc_train = pd.DataFrame(data=np.array(operator_acc_train), index=np.arange(len(operator_acc_train)))
    # operator_acc_train.to_csv('/ibgpu01/lmh/multi-omics/database/3.23/operator_acc_train_' + str(n) + '.tsv', sep='\t')
    #
    # operator_acc_val = pd.DataFrame(data=np.array(operator_acc_val), index=np.arange(len(operator_acc_val)))
    # operator_acc_val.to_csv('/ibgpu01/lmh/multi-omics/database/3.23/operator_acc_val_' + str(n) + '.tsv',
    #                           sep='\t')

    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(np.arange(len(train_loss)), np.array(train_loss), color='red', linestyle="solid", label="train loss")
    plt.plot(np.arange(len(train_loss_opt)), np.array(train_loss_opt), color='green', linestyle="solid", label="train loss(m_opt)")
    plt.plot(np.arange(len(val_loss_opt)), np.array(val_loss_opt), color='blue', linestyle="solid", label="val loss")
    plt.legend()
    plt.title('loss curve')
    plt.show()

    # train_loss = pd.DataFrame(data=np.array(train_loss), index=np.arange(len(train_loss)))
    # train_loss.to_csv('/ibgpu01/lmh/multi-omics/database/3.23/train_loss_' + str(n) + '.tsv', sep='\t')
    #
    # train_loss_opt = pd.DataFrame(data=np.array(train_loss_opt), index=np.arange(len(train_loss_opt)))
    # train_loss_opt.to_csv('/ibgpu01/lmh/multi-omics/database/3.23/train_loss_opt_' + str(n) + '.tsv', sep='\t')
    #
    # val_loss_opt = pd.DataFrame(data=np.array(val_loss_opt), index=np.arange(len(val_loss_opt)))
    # val_loss_opt.to_csv('/ibgpu01/lmh/multi-omics/database/3.23/val_loss_opt_' + str(n) + '.tsv', sep='\t')



if __name__ == "__main__":
    for i in range(1, 5+1):
        train_fs_ite(i=i, dataset='exp')
