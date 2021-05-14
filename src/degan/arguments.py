import argparse


def parse_arguments():

    parser = argparse.ArgumentParser(description='Audio Classification models')

    # Add arguments
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false', help='do not use cuda')
    parser.add_argument('--partition_path', default='partitions/partition_urbansound_fold10/', help='path to the partition folder containing the train, dev and test dataframes')
    
    parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='Batch size to use')
    parser.add_argument('--arc', default='lenet', help='network architecture: lenet')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs of no loss improvement before stop training')
    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='WD', help='weight decay for the optimizer')

    # GAN
    parser.add_argument('--critic_iters', type=int, default=5, metavar='CI', help='generator critic iterations')
    parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=10.0, help='gradient penalty regularization factor')

    # Store results
    parser.add_argument('--checkpoint', default='models/checkpoints', metavar='CHECKPOINT', help='checkpoints directory')
    parser.add_argument('--experiment_description', default='frist experiment', help='experiment name of the folder where best models are saved at')

    # Audio
    parser.add_argument('--sampling_rate', type=float, default=44100, metavar='SR', help='sampling rate of the audio signals')
    parser.add_argument('--time_window', type=float, default=0.37151927437641724, metavar='TW', help='time window covered by every data sample')

    return parser.parse_args()