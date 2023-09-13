import argparse


def get_args_avbae():
     parser = argparse.ArgumentParser(description="General Training Pipeline-AVBAE")
     parser.add_argument('--data_path', type=str, required=True, help='root path of dataset')
     parser.add_argument('--save_path', type=str, required=True, help='saved file path')
     parser.add_argument('--epochs', type=int, default=50)
     parser.add_argument('--lr_backbone', type=float, default=0.001, help='learning rate of parameters among encoder and decoder')
     parser.add_argument('--lr_dis', type=float, default=0.001, help='learning rate of parameters in discriminator')
     parser.add_argument('--batch_size', type=int, default=16)
     args = parser.parse_args()
     return args

def get_args_preprocess():
     parser = argparse.ArgumentParser(description='Preprocessing Dataset')
     parser.add_argument('--data_path', type=str, default='./data/', help='root path of downloaded dataset')
     parser.add_argument('--save_path', type=str, default='./data/h5/', help='saved file path')
     parser.add_argument('--dataset', type=str, required=True, choices=['exp','met', 'both'])


def get_args_modfr():
     parser = argparse.ArgumentParser(description="General Training Pipeline-MODFR")
     parser.add_argument('--data_path', type=str, default='./data/', help='root path of dataset')
     parser.add_argument('--save_path', type=str, required=True, help='saved file path')
     parser.add_argument('--pretrained_encoder', default='./data/checkpoints/opt_DI.pt',type=str, help='pretrained parameters of encoder')
     parser.add_argument('--use_pr', action='store_true', help='whether use pretrained encoder or not')
     parser.add_argument('--E1', type=int, default=300, help='updated steps on random masksets')
     parser.add_argument('--epochs', type=int, default=300)
     parser.add_argument('--batch_size', type=int, default=128, help='|D|')
     parser.add_argument('--mask_size', type=int, default=128, help='|M|')
     parser.add_argument('--f', type=float, default=0.5)
     parser.add_argument('--p', type=int, default=5)
     parser.add_argument('--random_mask_size', default=500, type=int)
     parser.add_argument('--FS', type=int, default=50)
     args = parser.parse_args()
     return args



