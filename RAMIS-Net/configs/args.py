import argparse


def get_args():
    """Parse and return command line arguments for RAMIS-Net training."""
    parser = argparse.ArgumentParser(description='RAMIS-Net')

    # Task and paths
    parser.add_argument('--task_name', type=str, default='RAMIS-Net', help='task name')
    parser.add_argument('--path_to_log', type=str, default='./new_results/', help='path to save results')
    parser.add_argument('--datasets', type=str, default='BRATS 2020', help='the type of dataset [BRATS 2018, BRATS 2020]')
    parser.add_argument('--path_to_data', type=str, default='./datasets/MICCAI_BraTS2020_TrainingData/', help='path to dataset')

    # Loss function parameters
    parser.add_argument('--context_loss', type=str, default='cosine', help='context loss type [cosine, L1]')
    parser.add_argument('--context_loss_l1_coef', type=float, default=0.2, help='RARA Loss level 1 weight')
    parser.add_argument('--context_loss_l2_coef', type=float, default=0.3, help='RARA Loss level 2 weight')
    parser.add_argument('--context_loss_l3_coef', type=float, default=0.4, help='RARA Loss level 3 weight')
    parser.add_argument('--context_loss_l4_coef', type=float, default=0.5, help='RARA Loss level 4 weight')
    parser.add_argument('--context_loss_full_coef', type=float, default=0.2, help='RARA Loss full weight')
    parser.add_argument('--token_loss', type=str, default='L1', help='token loss type [L1, MSE]')
    parser.add_argument('--token_loss_coef', type=float, default=0.1, help='MISA token loss weight')
    parser.add_argument('--recon_loss', type=str, default='L1', help='reconstruction loss')
    parser.add_argument('--recon_loss_coef', type=float, default=0.05, help='reconstruction loss weight')
    parser.add_argument('--consistency_coef', type=float, default=1.0, help='consistency loss weight')
    parser.add_argument('--weight_full_coef', type=float, default=0.4, help='full modality Dice weight')
    parser.add_argument('--weight_missing_coef', type=float, default=0.6, help='missing modality Dice weight')

    # Model and data parameters
    parser.add_argument('--modalities', type=str, nargs='*', default=['flair', 't1', 't1ce', 't2'],
                        help='modalities for training and evaluation')
    parser.add_argument('--n_missing_modalities', type=int, default=1, help='number of missing modalities')
    parser.add_argument('--number_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--batch_size_tr', type=int, default=1, help='training batch size')
    parser.add_argument('--batch_size_va', type=int, default=1, help='validation batch size')
    parser.add_argument('--progress_p', type=float, default=0.1, help='progress report frequency')
    parser.add_argument('--inputshape', default=[128, 160, 192], help='input shape')

    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--power', type=float, default=0.75, help='polynomial decay power')

    # Channel parameters
    parser.add_argument('--missing_in_chans', type=int, default=1, help='missing modality channels')
    parser.add_argument('--full_in_chans', type=int, default=4, help='full modality channels')

    args = parser.parse_args()
    return args
