import torch
import torch.nn as nn
from scipy import io as sio
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import random
from torch.nn.parallel import DataParallel
import time
from sklearn.metrics import confusion_matrix

from model import Encoder_Base,Encoder_FourLayer,Encoder_SixLayer
from model import Decoder, ACRNet  
from utils import load_real_data, load_synthesis_data,load_PaviaU_data,training_input_fn,initNetParams
from utils import get_args_forUnmixingData,get_args_forPaviaU
from utils import plotEndmembersAndGT, SAD_loss, MSE,write2txt,plot_loss,adjust_w_ab
from utils import DrawResult, CalAccuracy

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(2)


def setup_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def train_model(args, model,data_loader,optimizer_en,optimizer_de, iter_num, use_multiGPUs=False):
    L, L1, L2, L3, L4 = [], [], [], [], []
    W_ab = adjust_w_ab(args.init_w_ab, args.last_w_ab, args.epochs)  
    model.cuda()
    model.train()
    MSE_loss = nn.MSELoss(reduce=True, size_average=True)
    for _epoch in range(0, args.epochs): #
        l, l1, l2, l3, l4 = 0, 0, 0, 0, 0
        w_ab = W_ab[_epoch]
        for i, input in enumerate(data_loader):
            input= input[0].cuda()
            abunds, recon_x = model(input)  
            loss1 = args.w_sad *SAD_loss(recon_x,input)  # [batch_size, channel, H,W]
            loss2 = 0.0 * MSE_loss(recon_x,input)
            loss3 = w_ab * torch.sqrt(abunds).mean()
            if use_multiGPUs:
                em = model.module.decoder.decoder_weight.squeeze() # multiple GPUs
            else:
                em = model.decoder.decoder_weight.squeeze()
            z1 = em[:,:,:-1]
            z2 = em[:, :, 1:]
            em_loss = 0.5 * torch.sum((z1 - z2) * (z1 - z2))/z1.shape[0]
            loss = loss1 + loss2 + loss3 + em_loss

            optimizer_en.zero_grad()
            optimizer_de.zero_grad()
            loss.backward()

            optimizer_en.step()
            optimizer_de.step()

            l += loss.item()
            l1 += loss1.item()
            l2 += loss2.item()
            l3 += loss3.item()
            l4 += em_loss.item()
        l = l / iter_num
        l1 = l1 / iter_num
        l2 = l2 / iter_num
        l3 = l3 / iter_num
        l4 = l4 / iter_num
        L1.append(l1)
        L2.append(l2)
        L3.append(l3)
        L4.append(l4)
        L.append(l)
        if _epoch % 40 == 0:
            print('epoch [{}/{}],train loss:{:.4f}, MSE:{:.4f}, SAD:{:.4f}, 1/2 norm:{:.4f}, em:{:.4f}'
                  .format(_epoch + 1, args.epochs, l, l2, l1, l3, l4))
    print('---training is successfully down---')
    plot_loss(args.path, _epoch + 1, L, L1, L3, L4, args.Dataset)
    return model

# for real and synthetic datasets
def maincode_Unmixing_Dataset(data_path, Dataset, SNR, num_runs=1, save_file=False):
    start = time.time()
    use_multiGPUs = False
    if save_file:
        result_path = './result/repeatNums_'+str(num_runs)
        if 'synthetic' in Dataset:
            result_path = os.path.join(result_path, Dataset + '_' + SNR)
        else:
            result_path = os.path.join(result_path, Dataset)
        os.makedirs(result_path, exist_ok=True)
    
    args = get_args_forUnmixingData(Dataset,data_path,SNR)
    args.path=result_path

    if 'synthetic' in Dataset:
        img_3d, endmember_GT, abundance_GT, init_em = load_synthesis_data(args)
    else:
        img_3d, endmember_GT, abundance_GT, init_em = load_real_data(args)
    
    SAD_repeat, MSE_repeat = [], []
    endmember_repeat, abundance_repeat = [], []
    for i in range(num_runs):
        setup_seed(i+1)
        patches = training_input_fn(img_3d, args.patch_size, args.num_patches)
        train_dataset = TensorDataset(torch.tensor(patches, dtype=torch.float32))
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=True)
        
        mode = args.mode
        print('mode:', mode)
        if mode == 'base':
            encoder = Encoder_Base(args)  # deeper MultiHead feature for decoder
        elif mode == 'four_layer':
            encoder = Encoder_FourLayer(args)  # deeper MultiHead feature for decoder
        elif mode == 'six_layer':
            encoder = Encoder_SixLayer(args)

        decoder = Decoder(args=args, dim=args.Layernum1[2], qkv_PROJ=True, qkv_bias=False, kernel_size=args.kernel)
        model = ACRNet(encoder, decoder)
        model.apply(initNetParams)
        model_dict = model.state_dict()
        model_dict['decoder.decoder_weight'][args.kernel * args.kernel // 2, :, :] = torch.from_numpy(
            init_em.T).float()
        model.load_state_dict(model_dict)

        optimizer_en = torch.optim.RMSprop(model.encoder.parameters(), lr=args.lr_en, weight_decay=args.weight_decay)
        optimizer_de = torch.optim.RMSprop(model.decoder.parameters(), lr=args.lr_de, weight_decay=args.weight_decay)
        iter_num = patches.shape[0] // args.batch_size
        if use_multiGPUs:
            device_ids = list(range(torch.cuda.device_count()))
            print('device_ids:', device_ids)
            model = DataParallel(model, device_ids=device_ids).to(device_ids[0])

        model = train_model(args, model, data_loader, optimizer_en,
                                 optimizer_de, iter_num,use_multiGPUs)

        model.eval()
        hsi = np.transpose(img_3d, [2, 0, 1])  # Band, H,W
        hsi = torch.tensor(hsi, dtype=torch.float32).unsqueeze(0).cuda()
        # multiple GPUs for training
        if use_multiGPUs:
            endmembers = model.module.getEndmembers().detach().cpu().numpy()  # [ num_em,band]  # [band, num_em]
            abundances = model.module.getAbundances(hsi).detach().cpu().numpy()  # [num_em, H,W]
        else:
            endmembers = model.getEndmembers().detach().cpu().numpy()  # [ num_em,band]  # [band, num_em]
            abundances = model.getAbundances(hsi).detach().cpu().numpy()  # [num_em, H,W]
        SAD_ordered, endmember_sordered, abundance_sordered = plotEndmembersAndGT(endmembers, endmember_GT,
                                                                                         abundances)
        mse = MSE(abundance_GT, abundance_sordered)  # y_true, y_pred
        SAD_repeat.append(SAD_ordered)
        endmember_repeat.append(endmember_sordered)
        abundance_repeat.append(abundance_sordered)
        MSE_repeat.append(mse)
        print(f'the{i}th run of Dataset{Dataset}: ')
        print("SAD_repeat_i:[SAD_i, avg_SAD]", '\n', SAD_ordered)
        print("MSE_repeat_i:[mse_i, avg_mse]", '\n', mse)

        del model, decoder, encoder

    SAD_repeat=np.array(SAD_repeat)
    MSE_repeat = np.array(MSE_repeat)
    SAD_repeat_mean, SAD_repeat_std = SAD_repeat.mean(0), SAD_repeat.std(0)
    MSE_repeat_mean, MSE_repeat_std = MSE_repeat.mean(0), MSE_repeat.std(0)
    endmember_repeat = np.array(endmember_repeat)  # [num_runs, num_em, band]
    abundance_repeat = np.array(abundance_repeat)  # [num_runs, num_em, H,W]

    if save_file:
        filename = '/Proposed_' + Dataset + args.mode+'_Numpat_' + str(args.num_patches)+ '_epo_' + str(args.epochs) \
                   + '_bat_' + str(args.batch_size) + '_lr_en_' + str(args.lr_en)+ '_lr_de_' + str(args.lr_de) + '_numRuns_' + str(num_runs)
        result_filename = filename +'.mat'
        result_filename = result_path + result_filename
        print('result_filename save path:', '\n', result_filename)
        sio.savemat(result_filename,
                    {'endmember_repeat': endmember_repeat, 'abundance_repeat': abundance_repeat,
                     'SAD_repeat': SAD_repeat, 'MSE_repeat': MSE_repeat,
                     'SAD_repeat_mean': SAD_repeat_mean, 'SAD_repeat_std': SAD_repeat_std,
                     'MSE_repeat_mean': MSE_repeat_mean, 'MSE_repeat_std': MSE_repeat_std})

        assess_filename = result_path + filename +'.txt'
        write2txt(assess_filename, SAD_repeat, MSE_repeat, SAD_repeat_mean, SAD_repeat_std, MSE_repeat_mean, MSE_repeat_std)
    end = time.time()
    one_run_time = (end - start) / num_runs
    print('one_run_time: ', one_run_time)
    if save_file:
        file3 = open(assess_filename, 'a+', encoding='UTF-8')
        file3.write('one_run_time:' + str(one_run_time) + '\n')
        file3.close()
    print(f'the{num_runs} runs of Dataset{Dataset}: ')
    print("SAD_repeat_mean:[SAD_i, avg_SAD]", '\n', SAD_repeat_mean)
    print("MSE_repeat_mean:[mse_i, avg_mse]", '\n', MSE_repeat_mean)
    return SAD_repeat_mean[-1], MSE_repeat_mean[-1]

# Indiviual maincode for paviaU datset due to the shortage of abundance reference
def maincode_paviaU_Dataset(args, save_file=False):
    start = time.time()
    if save_file:
        result_path = './result'
        result_path = os.path.join(result_path, Dataset)
        os.makedirs(result_path, exist_ok=True)
    img_3d, classificationMap,spectrum,init_em=load_PaviaU_data(args)


    print('\n')
    setup_seed(1)
    patches = training_input_fn(img_3d, args.patch_size, args.num_patches)
    train_dataset = TensorDataset(torch.tensor(patches, dtype=torch.float32))
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        
    encoder = Encoder_SixLayer(args)
    decoder = Decoder(args=args, dim=args.Layernum1[2], qkv_PROJ=True, qkv_bias=False, kernel_size=args.kernel)
    model = ACRNet(encoder, decoder)
    model.apply(initNetParams)
    model_dict = model.state_dict()
    model_dict['decoder.decoder_weight'][args.kernel * args.kernel // 2, :, :] = torch.from_numpy(
        init_em.T).float()
    model.load_state_dict(model_dict)

    optimizer_en = torch.optim.RMSprop(model.encoder.parameters(), lr=args.lr_en, weight_decay=args.weight_decay)
    optimizer_de = torch.optim.RMSprop(model.decoder.parameters(), lr=args.lr_de, weight_decay=args.weight_decay)
    iter_num = patches.shape[0] // args.batch_size
 
    model = train_model(args, model, data_loader, optimizer_en,optimizer_de, iter_num)
    model.eval()

    hsi = np.transpose(img_3d, [2, 0, 1])
    hsi = torch.tensor(hsi, dtype=torch.float32).unsqueeze(0).cuda()
    endmembers = model.getEndmembers().detach().cpu().numpy()  # [ num_em,band]  # [band, num_em]
    abundances = model.getAbundances(hsi).detach().cpu().numpy()  # [num_em, H,W]

    SAD_ordered, endmember_sordered, abundance_sordered = plotEndmembersAndGT(endmembers, spectrum,
                                                                                     abundances)
    abundances = abundance_sordered

    print('SAD:', SAD_ordered)
    abundances_2d = np.reshape(abundances, [7, -1])
    idx = np.argmax(abundances_2d, axis=0)
    idx = idx + 1
    idx_2d = np.reshape(idx, [610, 340])
    val = classificationMap > 0
    val = val + 0
    idx_2d_val = idx_2d * val

    predictions = np.reshape(idx_2d_val, [1, -1]).squeeze()  
    references = np.reshape(classificationMap, [1, -1]).squeeze() 
    predictions_= [x for x in predictions if x>0]
    references_ = [x for x in references if x > 0]

    y_true =np.array(references_)
    y_pred = np.array(predictions_)
    oa, kappa, producer_acc=CalAccuracy(y_true, y_pred, n_classes=7)
    img = DrawResult(idx, 1)
    plt.imsave(result_path  + '/'+'paviaU.png', img)
    running_time = time.time() -start
    if save_file:
        result_file = os.path.join(result_path, 'PaviaU_result.mat')
        sio.savemat(result_file, {'prediction':idx_2d, 'visualization':img, 'pred_abundances':abundances})


        assess_filename = result_path  + '/PaviaU_classification.txt'
        file3 = open(assess_filename, 'w', encoding='UTF-8')
         # 'OA', 'Kappa', 'mode', 'num_patches', 'epochs', 'batch_size', 'lr_de', 'lr_en'\
        para_setting = 'mode, num_patches, epochs, batch_size, lr_de, lr_en'
        file3.write("para_setting" + '\n')
        file3.write(para_setting + '\n')
        result = args.mode+','+str(args.num_patches)+','+ str(args.epochs)+','+\
                 str(args.batch_size)+','+ str(args.lr_de)+','+str(args.lr_en)
        file3.write(result + '\n')
        file3.write('OA: '+str(oa) + '\n')
        file3.write('Kappa: ' + str(kappa) + '\n')
        for i in range(len(producer_acc)):
            file3.write(f"Class {i + 1} - producer_acc: {producer_acc[i]}"+ '\n')
        file3.write('running_time: ' + str(running_time) + '\n')
        file3.close()
    return  oa, kappa, producer_acc


if __name__ == '__main__':
    """-----------Real dataset------"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_path = '' # !!! replace it with your data_path 
    Avg_SAD, Avg_mse = [], []
    SNR = ['0db']
    num_runs = 1
    running_time = []
    Dataset = ['Muffle','Urban4','Samson'] 
    for dataset in Dataset:
        for snr in SNR:
            start = time.time()
            avg_SAD, avg_mse = maincode_Unmixing_Dataset(dataset, snr, num_runs, data_path, save_file=True)
            end = time.time()
            one_run_time = (end - start) / num_runs
            running_time.append(one_run_time)
            Avg_SAD.append(avg_SAD)
            Avg_mse.append(avg_mse)
   
    """-----------Synthetic dataset------"""
    # running_time_s = []
    # Avg_SAD_s, Avg_mse_s = [], []
    # num_runs = 1
    # SNR = ['20db', '30db']
    # Dataset= ['LMM_syntheticImage5', 'PPNMMM_syntheticImage5']
    # for dataset in Dataset:
    #     for snr in SNR:
    #         start = time.time()
    #         avg_SAD, avg_mse = maincode_Unmixing_Dataset(dataset, snr, num_runs, save_file=False)
    #         end = time.time()
    #         one_run_time = (end - start) / num_runs
    #         running_time_s.append(one_run_time)
    #         Avg_SAD_s.append(avg_SAD)
    #         Avg_mse_s.append(avg_mse)
    
    """-----------PaviaU dataset------"""
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # start = time.time()
    # Dataset = 'paviaU'
    # args =get_args_forPaviaU(Dataset, data_path)
    # oa, kappa, producer_acc = maincode_paviaU_Dataset(args, save_file=False)
    # end= time.time()
    # timecost = end - start
    # print('timecost:', timecost)