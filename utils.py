from torch import nn
import argparse
from scipy import io as sio
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import random
import matplotlib.pyplot as plt
import os
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math



def load_real_data(args) -> object:
    dataset = sio.loadmat(args.data_file)
    data, GT, abundance = dataset['img_3d'], dataset['endmember'], dataset['abundance']
    init_em = dataset['init_em']
    # (n_cols, n_rows, n_bands))
    data = data.transpose([1, 2, 0])
    n_rows, n_cols, n_bands = data.shape
    abundance = np.reshape(abundance, [abundance.shape[0],n_rows, n_cols])
    GT = GT.transpose([1, 0])


    print('data.shape:', data.shape)
    print('init endmember.shape:', init_em.shape)  # channel,num_em
    print('endmember.shape:', GT.shape)
    print('abundance.shape:', abundance.shape)
    return data, GT, abundance,init_em

def load_synthesis_data(args):
    # SNR='20db','30db','40db','50db'

    if args.Dataset=='LMM_syntheticImage5':
        img = 'syntheticImage5'
        em_name = 'endmember5'
        ab_name = 'abundance5'
    elif args.Dataset=='PPNMMM_syntheticImage5':
        img = 'syntheticImage5'
        em_name = 'endmember5'
        ab_name = 'abundance5'


    dataset = sio.loadmat(args.data_file)
    img_name = img + "_"+ args.SNR
    init_em_name = "endmember_" + args.SNR
    data, GT, abundance = dataset[img_name], dataset[em_name], dataset[ab_name]
    init_em = dataset[init_em_name]


    if 'synthetic' in dataset:
        # n_bands, n_rows, n_cols = data# C, H,W
        abundance = abundance.transpose([2, 0, 1]) # C, H,W
        GT = GT.transpose([1, 0])
    else:
        # (n_cols, n_rows, n_bands))
        data = data.transpose([1, 2, 0]) # H, W,C
        n_rows, n_cols, n_bands = data.shape
        abundance = np.reshape(abundance, [abundance.shape[0], n_rows, n_cols])# c, h,w
        GT = GT.transpose([1, 0])

    print('data.shape:', data.shape) # H, W,C
    print('init endmember.shape:', init_em.shape)  # C,num_em
    print('endmember.shape:', GT.shape) # C, num_em
    print('abundance.shape:', abundance.shape)# num_em, h,w
    return data, GT, abundance,init_em



def load_PaviaU_data(args):
    dataset = sio.loadmat(args.data_file)
    img_3d, classificationMap = dataset['img_3d'], dataset['ClassificationGT7']
    spectrum = dataset['spectrum_GT7']
    init_em = dataset['init_em']

    return img_3d, classificationMap,spectrum,init_em




def get_args_forUnmixingData(Dataset,data_path, SNR='20db'):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--Dataset', default=Dataset,
                        type=str, help='path filename of training data')
    parser.add_argument('--SNR', default=SNR, type=str, metavar='C', help='C')  # 0.5

    data_file, C, H, W, num_em = get_sample(Dataset, data_path)
    parser.add_argument('--data_file', default=data_file,
                        type=str, help='path filename of the trained model')
    dim1 = num_em * 16
    dim2 = num_em * 8
    layernum1 = [C, dim1, dim2, num_em]
    parser.add_argument('--Layernum1', default=layernum1, type=list, nargs='+', help='layernum')
    parser.add_argument('--head_num', default=num_em, type=int, metavar='C', help='C')  # 5
    parser.add_argument('--kernel', default=5, type=int, metavar='C', help='C')  # 5
    parser.add_argument('--scale', default=1, type=float, metavar='C', help='C')  # 0.5

    w_sad, w_mse, init_w_ab, last_w_ab = 1, 0, 0.5, 0 # 20240927
    parser.add_argument('--w_sad', default=w_sad, type=int, metavar='C', help='C')  # 0.5
    parser.add_argument('--init_w_ab', default=init_w_ab, type=float, metavar='C', help='C')  # 0.5
    parser.add_argument('--last_w_ab', default=last_w_ab, type=float, metavar='C', help='C')  # 0.5
    parser.add_argument('--patch_size', default=13, type=int, metavar='C', help='C') # 39;40

    if Dataset == 'Urban4':
        num_patches,epochs, batch_size, mode = 800, 200, 32,'base'
        lr_en, lr_de = 1e-3, 1e-3
    elif Dataset =='Muffle':
        num_patches, epochs, batch_size, mode =  800, 200, 32,'base'
        lr_en, lr_de = 1e-3, 5e-4
    elif Dataset == 'Samson':
        num_patches, epochs, batch_size, mode =  800, 200, 32,'base'
        lr_en, lr_de = 1e-3, 0.00015
    elif 'syntheticImage' in Dataset:
        num_patches, epochs, batch_size, mode = 800, 200, 32, 'four_layer' 
        lr_en, lr_de = 1e-3, 1e-5
    else:
        assert Dataset in ['Urban4',  'Samson', 'Muffle',
                           'LMM_syntheticImage5', 'PPNMMM_syntheticImage5'], "Dataset 出错了"


    parser.add_argument('--num_patches', default=num_patches, type=int, metavar='C', help='C') 
    parser.add_argument('--epochs', default=epochs, type=int, metavar='N', help='defalut:100')
    parser.add_argument('--mode', default=mode, type=str, help='path filename of the trained model')
    parser.add_argument('--batch_size', default=batch_size, type=int, metavar='C', help='C') 
    parser.add_argument('--lr_en',  default=lr_en, type=float, 
                        help='initial (base) learning rate')
    parser.add_argument('--lr_de',  default=lr_de, type=float, 
                        help='initial (base) learning rate')

    parser.add_argument('--channel', default=C, type=int, metavar='C', help='C')
    parser.add_argument('--H', default=H, type=int, metavar='H', help='H')
    parser.add_argument('--W', default=W, type=int, metavar='W', help='W')
    parser.add_argument('--num_em', default=num_em, type=int, metavar='C', help='C')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay') 

    args = parser.parse_args()
    print('Dataset:', args.Dataset)
    print('data_file:', args.data_file)
    print('encoder Layernum1:', args.Layernum1,' | decoder kernel:', args.kernel )
    print('args.init_w_ab:', args.init_w_ab, ' | args.last_w_ab:', args.last_w_ab)


    print('SNR:', args.SNR)
    print('patch_size:', args.patch_size, ' | num_patches:', args.num_patches,
          ' | batch_size:', args.batch_size, ' | epochs:', args.epochs)
    print('lr_en:  ', args.lr_en, '|  lr_de:', args.lr_de)
    print('weight_decay:', args.weight_decay)
    print('mode:', args.mode)
    return args

def get_args_forPaviaU(Dataset,data_path):
    print('---------------------------func: get_args_adjust_w_ab---------------------------')
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--Dataset', default=Dataset,
                        type=str, help='path filename of training data')


    data_file, C, H, W, num_em = get_sample(Dataset, data_path)
    parser.add_argument('--data_file', default=data_file,
                        type=str, help='path filename of the trained model')
    dim1 = num_em * 16
    dim2 = num_em * 8
    layernum1 = [C, dim1, dim2, num_em]
    parser.add_argument('--Layernum1', default=layernum1, type=list, nargs='+', help='layernum')
    parser.add_argument('--head_num', default=num_em, type=int, metavar='C', help='C')  
    parser.add_argument('--kernel', default=5, type=int, metavar='C', help='C') 
    parser.add_argument('--scale', default=1, type=float, metavar='C', help='C')  

    w_sad, w_mse, init_w_ab, last_w_ab = 1, 0, 0.5, 0 

    parser.add_argument('--w_sad', default=w_sad, type=int, metavar='C', help='C') 
    parser.add_argument('--w_mse', default=w_mse, type=int, metavar='C', help='C')  
    parser.add_argument('--beta', default=0, type=float, metavar='C', help='C') 
    parser.add_argument('--init_w_ab', default=init_w_ab, type=float, metavar='C', help='C')  
    parser.add_argument('--last_w_ab', default=last_w_ab, type=float, metavar='C', help='C')  

    patch_size = 13 
    parser.add_argument('--patch_size', default=patch_size, type=int, metavar='C', help='C') 

    parser.add_argument('--num_patches', default=800, type=int, metavar='C', help='C')  
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='defalut:100')
    parser.add_argument('--batch_size', default=32, type=int, metavar='C', help='C') 
    parser.add_argument('--lr_de',  default=1e-3, type=float, 
                        help='initial (base) learning rate')
    parser.add_argument('--mode', default='six_layer', type=str, help='')

    parser.add_argument('--lr_en',  default=1e-3, type=float, 
                        help='initial (base) learning rate')

    parser.add_argument('--channel', default=C, type=int, metavar='C', help='C')
    parser.add_argument('--H', default=H, type=int, metavar='H', help='H')
    parser.add_argument('--W', default=W, type=int, metavar='W', help='W')
    parser.add_argument('--num_em', default=num_em, type=int, metavar='C', help='C')

    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')


    weight_decay = 0.001

    parser.add_argument('--weight_decay', type=float, default=weight_decay, help='weight_decay') # 0.00001; 0.0001; 0.0005;0.001;0.01

    args = parser.parse_args()
    print('Dataset:', args.Dataset)
    print('data_file:', args.data_file)
    print('encoder Layernum1:', args.Layernum1,' | decoder kernel:', args.kernel )
    print('args.w_sad:', args.w_sad, ' | args.w_mse:', args.w_mse)
    print('args.init_w_ab:', args.init_w_ab, ' | args.last_w_ab:', args.last_w_ab)

    print('patch_size:', args.patch_size, ' | num_patches:', args.num_patches,
          ' | batch_size:', args.batch_size, ' | epochs:', args.epochs)
    print('lr_en:  ', args.lr_en, '|  lr_de:', args.lr_de)
    print('weight_decay:', args.weight_decay)
    return args

def training_input_fn(hsi, patch_size, patch_number):
    patches = extract_patches_2d(hsi, (patch_size, patch_size), max_patches=patch_number)
    patches = np.transpose(patches,[0,3,1,2])
    return patches

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)      # m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)      # init.constant(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)        # init.constant(m.bias, 0)



def get_sample(dataset, data_path):
    C,H, W,num_em = 0,0,0,0
    if dataset=='Urban4':
        name='Urban_188_em4_init.mat'
        C, H, W = 162, 307, 307
        num_em = 4
    elif dataset=='Samson':
        name = 'Samson1_156_em3_init.mat'
        C, H, W = 156, 95, 95
        num_em = 3
    elif dataset=='Muffle':
        name = 'Muffle_64_em5_init.mat'
        C, H, W = 64, 130, 90
        num_em = 5

    elif dataset=='LMM_syntheticImage5':
        C, H, W = 269, 256, 256
        num_em = 5
        name='LMMsyntheticImage5.mat'

    elif dataset=='PPNMMM_syntheticImage5':
        num_em = 5
        C, H, W = 269, 256, 256
        name = 'PPNMMM_syntheticImage5.mat'
    elif dataset=='paviaU':
        num_em = 7
        C, H, W = 103, 610, 340
        name = 'paviaU_unmixing7.mat'

    data_file = data_path + name
    return data_file, C,H, W, num_em



def plotEndmembersAndGT(endmembers, endmembersGT, abundances):
    # endmembers & endmembersGT:[num_em, band](4, 162)
    # Abundance: [num_em, H,W] (4, 307, 307)
    num_endmembers = endmembers.shape[0]
    n = num_endmembers // 2  # how many digits we will display
    if num_endmembers % 2 != 0: n = n + 1
    dict, SAD_, Average_SAM = obtain_endmembers_order(endmembers, endmembersGT)
    SAD_ordered = []
    endmember_sordered = []
    abundance_sordered = []


    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        endmember_sordered.append(endmembers[dict[i]])
        abundance_sordered.append(abundances[dict[i], :, :])
    endmember_sordered = np.array(endmember_sordered)
    for i in range(num_endmembers):
        z = numpy_SAD(endmember_sordered[i], endmembersGT[i, :])
        SAD_ordered.append(z)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "aSAM score for all endmembers: " + format(Average_SAM, '.3f') + " radians"
    st = plt.suptitle(title)
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembersGT[i, :], 'r', linewidth=1.0, label='GT')
        plt.plot(endmember_sordered[i, :], 'k-', linewidth=1.0, label='predict')
        ax.set_title("SAD: " + str(i) + " :" + format(SAD_ordered[i], '.4f'))
        ax.get_xaxis().set_visible(False)
    SAD_ordered.append(Average_SAM)
    SAD_ordered = np.array(SAD_ordered)
    abundance_sordered = np.array(abundance_sordered)  # [3, 95, 95]
    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)
    return SAD_ordered, endmember_sordered, abundance_sordered
def plotAbundances(abundances):
    num_endmembers = abundances.shape[0]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1

    fig = plt.figure(2, figsize=[8, 8])
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[i,:, :], cmap='viridis')
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im.set_clim([0, 1])
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

def numpy_SAD(y_true, y_pred):
    return np.arccos(y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))
def obtain_endmembers_order(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    SAD_ = []
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    print('sad_mat:', sad_mat)
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if minimum == 100:
            break
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        dict[index[1]] = index[0]  # keep Gt at first,
        SAD_.append(minimum)
        # dict[index[0]] = index[1]
        sad_mat[index[0], index[1]] = 100
        rows += 1
        sad_mat[index[0], :] = 100
        sad_mat[:, index[1]] = 100
    SAD_ = np.array(SAD_)
    Average_SAM = np.sum(SAD_)/ len(SAD_)
    return dict, SAD_, Average_SAM

def SAD_loss(y_true, y_pred):
    y_true = torch.nn.functional.normalize(y_true, dim=1, p=2) # [batch_size, band, patch_size,patch_size]
    y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=2)

    A = torch.mul(y_true, y_pred) # [batch_size, band, patch_size,patch_size]
    A = torch.sum(A, dim=1) # [batch_size, patch_size,patch_size]
    A = torch.clamp(A, -1.0, 1.0)
    sad = torch.acos(A) # [batch_size, patch_size,patch_size]
    loss = torch.mean(sad)
    return loss

def MSE(y_true, y_pred):
    # y_true:[num_em,H,W]
    # y_pred:[num_em,H,W]
    # MSE--相当于y-y_hat的二阶范数的平方/n
    # y_pred = np.transpose(y_pred, axes=[2, 0,1 ]) # [num_em,H,W]

    num_em = y_true.shape[0]
    y_true = np.reshape(y_true , [num_em, -1])
    y_pred = np.reshape(y_pred , [num_em, -1])
    # num = y_pred.shape[1]

    R = y_pred - y_true
    r = R*R
    mse = np.mean(r, axis=1)
    Average_mse = np.sum(mse) / len(mse)
    mse = np.insert(mse, num_em, Average_mse, axis=0)
    return mse
def write2txt(filename, SAD_repeat, MSE_repeat, SAD_mean,SAD_std,MSE_mean, MSE_std):
    print('save filename:', '\n',filename)
    file3 = open(filename, 'w', encoding='UTF-8')
    # SAD_mean
    file3.write('SAD_mean: #1, #2, #3, #4, mean' + '\n')
    file3.write(str(SAD_mean) + '\n')
    file3.write('SAD_std: #1, #2, #3, #4, mean' + '\n')
    file3.write(str(SAD_std) + '\n')

    file3.write('MSE_mean: #1, #2, #3, #4, mean' + '\n')
    file3.write(str(MSE_mean) + '\n')
    file3.write('MSE_std: #1, #2, #3, #4, mean' + '\n')
    file3.write(str(MSE_std) + '\n')

    # SAD_repeat
    num_runs = SAD_repeat.shape[0]
    file3.write('\n')
    file3.write('SAD_repeat: #1, #2, #3, #4, mean' + '\n')
    file3.write('num_runs:' +str(num_runs)+ '\n')
    file3.write(str(SAD_repeat) + '\n')

    # MSE_repeat
    # num_runs = SAD_repeat.shape[0]
    file3.write('\n')
    file3.write('MSE_repeat: #1, #2, #3, #4, mean' + '\n')
    file3.write('num_runs:' + str(num_runs) + '\n')
    file3.write(str(MSE_repeat) + '\n')
    file3.close()

def plot_loss(path, epochs, L_total, L1,L2,L3,name):
    plt.figure()
    plt.plot(range(epochs), L_total, label='Training loss', linewidth=2)
    plt.plot(range(epochs), L1, label='SAD loss', linestyle='--', linewidth=2)
    plt.plot(range(epochs), L2, label='L1/2 Norm loss', linestyle='-.', linewidth=2)
    plt.plot(range(epochs), L3, label='Smooth loss', linestyle=':', linewidth=2)


    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim((0, epochs))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)


    plt.tight_layout()
    plt.savefig(path+name+'_loss.png')
    plt.show()
def adjust_w_ab(init_w_ab, last_w_ab, epochs):
    W_ab = []
    if init_w_ab > last_w_ab:
        gap = init_w_ab - last_w_ab
        for cur_epoch in range(epochs):
            cur_w_ab = gap * 0.5 * (1. + math.cos(math.pi * cur_epoch / epochs)) + last_w_ab
            W_ab.append(cur_w_ab)
    else:
        gap = last_w_ab - init_w_ab
        for cur_epoch in range(epochs):
            cur_w_ab = gap * 0.5 * (1. + math.cos(-math.pi * (epochs-cur_epoch) / epochs)) + init_w_ab
            W_ab.append(cur_w_ab)
    W_ab = np.array(W_ab)

    return W_ab










def DrawResult(labels, imageID):
    # ID=1: Pavia University
    # ID=2: Salinas
    # ID=3: Houston

    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(1, num_class + 1):
        X_result[np.where(labels == i), 0] = palette[i - 1, 0]
        X_result[np.where(labels == i), 1] = palette[i - 1, 1]
        X_result[np.where(labels == i), 2] = palette[i - 1, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result
def CalAccuracy(y_true, y_pred, n_classes = 7):
    # prediction: 1,2,...,7;  label: 1,2,...,7;
    def confusion_matrix(y_true, y_pred, n_classes):
        cm = np.zeros((n_classes, n_classes))
        for i in range(len(y_true)):
            cm[y_true[i] - 1][y_pred[i] - 1] += 1
        return cm
    def overall_accuracy(cm):
        total_sum = np.sum(cm)
        correct_sum = np.trace(cm)
        return correct_sum / total_sum
    def kappa_coefficient(cm):
        total_sum = np.sum(cm)
        observed_accuracy = overall_accuracy(cm)
        expected_accuracy = 0
        for i in range(cm.shape[0]):
            row_sum = np.sum(cm[i])
            col_sum = np.sum(cm[:, i])
            expected_accuracy += row_sum * col_sum / total_sum ** 2
        return (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
    def class_accuracy_and_kappa(cm):
        n_classes = cm.shape[0]
        class_accuracies = []
        class_kappas = []
        for i in range(n_classes):
            true_positives = cm[i][i]
            total_true = np.sum(cm[i])
            total_predicted = np.sum(cm[:, i])
            if total_true == 0 and total_predicted == 0:
                class_accuracies.append(0)
            else:
                class_oa = true_positives / (total_true + total_predicted - true_positives)
                class_oa = round(class_oa, 4)
                class_accuracies.append(class_oa)
            class_kappa = 0
            if total_true > 0 and total_predicted > 0:
                observed_accuracy = true_positives / (total_true + total_predicted - true_positives)
                expected_accuracy = ((np.sum(cm[i]) * np.sum(cm[:, i])) / (np.sum(cm) ** 2))
                class_kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
                class_kappa = round(class_kappa, 4)
            class_kappas.append(class_kappa)
        return class_accuracies, class_kappas

    def producer_accuracy(cm):
        n_classes = cm.shape[0]
        producer_accuracies = []
        for i in range(n_classes):
            true_positives = cm[i][i]
            total_true = np.sum(cm[i])
            if total_true > 0:
                producer_a = true_positives / total_true
                producer_a = round(producer_a, 4)
                producer_accuracies.append(producer_a)
            else:
                producer_accuracies.append(0)
        return producer_accuracies

    cm = confusion_matrix(y_true, y_pred, n_classes)
    oa = overall_accuracy(cm)
    kappa = kappa_coefficient(cm)
    # class_oa, class_kappas = class_accuracy_and_kappa(cm)
    producer_acc = producer_accuracy(cm)
    oa = round(oa, 4)
    kappa = round(kappa, 4)
    print(f"Overall Accuracy (OA): {oa}")
    print(f"Overall Kappa: {kappa}")
    for i in range(n_classes):
        print(f"Class {i + 1} - producer_acc: {producer_acc[i]}")
        # print(f"Class {i + 1} - Accuracy: {class_oa[i]}, Kappa: {class_kappas[i]}")
    return oa, kappa, producer_acc
