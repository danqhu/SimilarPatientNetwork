import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import InterpolationMode
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from preprocessing_pT import show_slices
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, average_precision_score, classification_report
import math
import platform
import pickle
import pandas as pd
import scipy
# plt.ion()  # interactive mode

IMG_EXTENSIONS = ('.pt', '.npy')



'''
custum functions
'''
def img_tensor_loader(path:str) -> Any:
    return torch.load(path)
def img_numpy_loader(path:str) -> Any:
    return np.load(path)


def save_dict(dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict


class MyDatasetFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(MyDatasetFolder, self).__init__(root=root,
                                              loader=loader,
                                              extensions=extensions,
                                              transform=transform,
                                              target_transform=target_transform,
                                              is_valid_file=is_valid_file)

    '''
    重写__getitem__方法，使得可以得到每个文件的文件名，用于Tumor level判断
    '''
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if type(sample) is not tuple:
            sample = (sample,)

        return sample, target, path


def MyCollate_fn(batch):
    data = []
    target = []
    file_path = []

    #     for item in batch:
    #         data.append(item[0])
    #         target.append(item[1])  # image labels.
    #         file_path.append(item[2])

    for item in batch:
        for image in item[0]:
            data.append(image)
            target.append(item[1])  # image labels.
            file_path.append(item[2])
    target = torch.LongTensor(target)  # image labels.
    return data, target, file_path


'''
实例function
'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def set_parameter_requires_grad(model, num_frozen_layer):
    if num_frozen_layer < 0:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for i, param in enumerate(model.parameters()):
            if i < num_frozen_layer:
                param.requires_grad = False

def initialize_model(model_name, num_classes, num_slices, resnet_type = '101', dropout=0.1, num_frozen_layer=-1, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0



    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        num_ftrs = model_ft.fc.in_features
        num_input_channels = model_ft.conv1.in_channels
        if num_input_channels != num_slices:
            layer1_input_channels = num_slices
            layer1_output_channels = model_ft.conv1.out_channels
            layer1_kernel_size = model_ft.conv1.kernel_size[0]
            layer1_stride = model_ft.conv1.stride[0]
            layer1_padding = model_ft.conv1.padding[0]
            model_ft.conv1 = nn.Conv2d(layer1_input_channels, layer1_output_channels, layer1_kernel_size, layer1_stride, layer1_padding, bias=False)

        # if use_pretrained:
        #     state_dict = models.resnet.load_state_dict_from_url(models.resnet.model_urls['resnet101'], progress=True)
        #     model_ft.load_state_dict(state_dict, strict=False)
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet3D":
        """r3d_18
        """
        model_ft = resnet3D(num_classes=num_classes, num_frozen_layer=num_frozen_layer)


    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, num_frozen_layer)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "resnet_bilstm":

        model_ft = resnet_bilstm(num_classes=num_classes, num_frozen_layer=num_frozen_layer, bidirectional=True)
        input_size = 224

    elif model_name == "resnet_transformer":

        model_ft = resnet_transformer(
            num_classes=num_classes,
            num_frozen_layer=num_frozen_layer,
            resnet_type=resnet_type,
            hidden_feature_size=128,
        )
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def make_decision(epoch_metrics, dataset_sizes, phase,
                  epoch_pred_proba, epoch_labels,
                  epoch_file_path: Optional[list] = None,
                  decision_level='Slices', decision_strategy='mean'):

    # if decision_level == 'Tumor':
    #     probas = {}
    #     labels = {}
    #
    #     for i, item in enumerate(epoch_file_path):
    #         if 'aug' in item:
    #             continue
    #         file_name = item.replace("\\","/").split("/")   # 为了兼容windows和linux
    #         label = file_name[-2]
    #         patient_ID = file_name[-1].split("_")[0]+"_"+file_name[-1].split("_")[1]
    #         if patient_ID in probas and patient_ID in labels:
    #             probas[patient_ID].append(epoch_pred_proba[i])
    #             if labels[patient_ID] != epoch_labels[i]:
    #                 raise ValueError("The labels are inconsistant between two slices of one tumor!")
    #         else:
    #             probas[patient_ID] = [epoch_pred_proba[i]]
    #             labels[patient_ID] = epoch_labels[i]
    #
    #     probas_list = []
    #     labels_list = []
    #
    #     # if decision_strategy == "mode":
    #     #     strategy = lambda x: scipy.stats.mode([np.argmx(pred) for pred in x])
    #     # elif decision_strategy == "max":
    #     #     strategy = lambda x: np.max([np.argmax(pred) for pred in x])
    #     #
    #     # for key in probas.keys():
    #     #     for prob in probas[key]:
    #     #         probas_list.append(strategy(probas[key]))
    #     #         labels_list.append(labels[key])
    #     for key in probas.keys():
    #         probas_list.append(probas[key])
    #         labels_list.append(labels[key])
    #
    #     probas_arr = np.array(probas_list)
    #     labels_arr = np.array(labels_list)
    #
    #     epoch_metrics['AUC'] = roc_auc_score(y_true=labels_arr, y_score=probas_arr)
    #     epoch_metrics['AUPRC'] = average_precision_score(y_true=labels_arr, y_score=probas_arr)
    #     precisions, recalls, thresholds = precision_recall_curve(y_true=labels_arr, probas_pred=probas_arr)
    #
    #
    #     threshold = thresholds[math.floor(thresholds.shape[0]/2)]
    #     epoch_pred_labels = np.where(probas_arr < threshold, 0, 1).astype(np.int)
    #     epoch_f1 = f1_score(y_true=labels_arr, y_pred=epoch_pred_labels)
    #     epoch_confusion_matrix = confusion_matrix(y_true=labels_arr, y_pred=epoch_pred_labels)
    #     epoch_metrics['ACC'] = np.sum(labels_list == epoch_pred_labels)/len(labels_list)
    #     temp = 1
    #
    # elif decision_level == 'Slices':
    #     epoch_metrics['ACC'] = running_corrects.item() / dataset_sizes[phase]
    #     epoch_metrics['AUC'] = roc_auc_score(y_true=epoch_labels, y_score=epoch_pred_proba)
    #     epoch_metrics['AUPRC'] = average_precision_score(y_true=epoch_labels, y_score=epoch_pred_proba)
    #     precisions, recalls, thresholds = precision_recall_curve(y_true=epoch_labels, probas_pred=epoch_pred_proba)
    #     threshold = thresholds[math.floor(thresholds.shape[0]/2)]
    #     epoch_pred_labels = np.where(epoch_pred_proba < threshold, 0, 1).astype(np.int)
    #     epoch_f1 = f1_score(y_true=epoch_labels, y_pred=epoch_pred_labels)
    #     epoch_confusion_matrix = confusion_matrix(y_true=epoch_labels, y_pred=epoch_pred_labels)
    #     probas_arr = np.array(epoch_pred_proba)
    #     labels_arr = np.array(epoch_labels)
    # else:
    #     raise ValueError('No proper devision level assigned ! ')
    epoch_metrics['AUC'] = roc_auc_score(y_true=epoch_labels, y_score=epoch_pred_proba, multi_class='ovr') if is_pT else roc_auc_score(y_true=epoch_labels, y_score=epoch_pred_proba[:,1])
    # epoch_metrics['AUPRC'] = average_precision_score(y_true=epoch_labels, y_score=epoch_pred_proba)

    # precisions, recalls, thresholds = precision_recall_curve(y_true=labels_arr, probas_pred=probas_arr)



    epoch_pred_labels = np.argmax(epoch_pred_proba, axis=1)
    epoch_metrics['AUPRC'] = f1_score(y_true=epoch_labels, y_pred=epoch_pred_labels, average='macro') if is_pT else average_precision_score(y_true=epoch_labels, y_score=epoch_pred_proba[:,1])
    epoch_confusion_matrix = confusion_matrix(y_true=epoch_labels, y_pred=epoch_pred_labels)
    epoch_metrics['ACC'] = np.sum(epoch_labels == epoch_pred_labels)/len(epoch_labels)
    # print(epoch_f1)
    print(epoch_confusion_matrix)
    # print(classification_report(y_true=epoch_labels, y_pred=epoch_pred_labels))

    return epoch_metrics, epoch_pred_proba, epoch_pred_labels

def _my_stack(crops):
    return torch.stack([crop for crop in crops])

def _my_crops(if_crop, input_size):
    if if_crop == 'CenterCrop':
        return transforms.CenterCrop(input_size)
    elif if_crop == 'FiveCrop':
        return transforms.FiveCrop(input_size)
    elif if_crop == 'TenCrop':
        return transforms.TenCrop(input_size)
    else:
        return None




class resnet_bilstm(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 hidden_feature_size=512,
                 hidden_size_lstm=128,
                 bidirectional=False):
        super(resnet_bilstm, self).__init__()
        self.num_classes = num_classes
        self.hidden_feature_size = hidden_feature_size
        self.hidden_size_lstm = hidden_size_lstm

        self.resnet = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(self.resnet, num_frozen_layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=self.hidden_feature_size)


        self.lstm = nn.LSTM(input_size=self.hidden_feature_size, hidden_size=self.hidden_size_lstm, bidirectional=bidirectional)
        self.fc1 = nn.Linear(in_features=self.hidden_size_lstm*2 if bidirectional else self.hidden_size_lstm, out_features=self.num_classes)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)



    def forward(self, x_images):

        deep_features = []
        for x_image in x_images:
            x_image = self.resnet.conv1(x_image)
            x_image = self.resnet.bn1(x_image)
            x_image = self.resnet.relu(x_image)
            x_image = self.resnet.maxpool(x_image)

            x_image = self.resnet.layer1(x_image)
            x_image = self.resnet.layer2(x_image)
            x_image = self.resnet.layer3(x_image)
            x_image = self.resnet.layer4(x_image)

            x_image = self.resnet.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet.fc(x_image)
            deep_features.append(x_image)

        deep_features_pack = pack_sequence(sequences=deep_features, enforce_sorted=False)

        outputs, (hn, cn) = self.lstm(deep_features_pack)
        hn = hn.squeeze(dim=0)
        hn_concat = torch.cat((hn[0,:,:], hn[1,:,:]), dim=1)
        outputs = self.fc1(hn_concat)
        # outputs = self.softmax(outputs)


        '''
        为了测试经过pack_sequence之后LSTM的输出hn是否还是batch中原始顺序输出的
        事实证明是的，因此可直接在hn基础上进行预测分类，并能够和labels对应
        '''
        # outputs_unpacked, lens_unpacked = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0)
        # temp = outputs_unpacked[0,lens_unpacked[0]-1,:]
        # temp2 = hn[0,0,:]
        # temp3 = outputs_unpacked[0,0,128:]
        # temp4 = hn[1,0,:]
        # '''经验证，output里面正向和反向concat是在相对称的时间戳上面，例如正向在t=1,负向的在t=n-1'''
        # print(outputs)
        return outputs

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class resnet_transformer(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 hidden_feature_size=128,
                 dropout = 0.1,
                 resnet_type='101'
                 ):
        super(resnet_transformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_feature_size = hidden_feature_size

        if resnet_type == '101':
            self.resnet = models.resnet101(pretrained=use_pretrained)
        elif resnet_type == '50':
            self.resnet = models.resnet50(pretrained=use_pretrained)
        elif resnet_type == '34':
            self.resnet = models.resnet34(pretrained=use_pretrained)
        elif resnet_type == '18':
            self.resnet = models.resnet18(pretrained=use_pretrained)
        else:
            Exception('No proper resnet type set!')
        set_parameter_requires_grad(self.resnet, num_frozen_layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=self.hidden_feature_size)

        self.pos_encoder = PositionalEncoding(d_model=self.hidden_feature_size, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2)

        self.fc2 = nn.Linear(in_features=self.hidden_feature_size, out_features=8)
        self.fc1 = nn.Linear(in_features=8, out_features=self.num_classes)




    def forward(self, x_images):

        deep_features = []
        for x_image in x_images:
            x_image = self.resnet.conv1(x_image)
            x_image = self.resnet.bn1(x_image)
            x_image = self.resnet.relu(x_image)
            x_image = self.resnet.maxpool(x_image)

            x_image = self.resnet.layer1(x_image)
            x_image = self.resnet.layer2(x_image)
            x_image = self.resnet.layer3(x_image)
            x_image = self.resnet.layer4(x_image)

            x_image = self.resnet.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet.fc(x_image)
            deep_features.append(x_image)

        '''
        为了测试经过pack_sequence之后LSTM的输出hn是否还是batch中原始顺序输出的
        事实证明是的，因此可直接在hn基础上进行预测分类，并能够和labels对应
        '''
        deep_features_pack = pack_sequence(sequences=deep_features, enforce_sorted=False)
        # outputs, (hn, cn) = self.lstm(deep_features_pack)
        outputs_unpacked, lens_unpacked = pad_packed_sequence(deep_features_pack, batch_first=False, padding_value=0.0)

        #
        # # '''添加标志位'''
        # # cls = torch.zeros((1, n, e)).to(device)
        # # cls[0, :, 0] = 1.0
        # # outputs_unpacked = torch.cat((cls, outputs_unpacked), 0)
        # # s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        # # src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        # # for i in range(lens_unpacked.shape[0]):
        # #     src_key_padding_mask[i, lens_unpacked[i]+1:] = True
        #
        s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        h_s = torch.zeros((n, e)).to(device)
        for i in range(n):
            temp_output = outputs_trans[:lens_unpacked[i], i, :]
            h_s[i, :] = torch.mean(temp_output, dim=0)

        h_s = self.fc2(h_s)

        outputs = self.fc1(h_s)

        # outputs = self.softmax(outputs)

        # temp = outputs_unpacked[0,lens_unpacked[0]-1,:]
        # temp2 = hn[0,0,:]
        return outputs, h_s

class resnet3D(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 ):
        super(resnet3D, self).__init__()
        self.num_classes = num_classes

        """r3d_18
        """
        self.resnet3D = models.video.r3d_18(pretrained=use_pretrained)
        set_parameter_requires_grad(self.resnet3D, num_frozen_layer)

        # customStem = nn.Sequential(
        #     nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.resnet3D.stem = customStem
        num_ftrs = self.resnet3D.fc.in_features
        self.resnet3D.fc = nn.Linear(num_ftrs, num_classes)



    def forward(self, x_images):

        deep_features = []
        for x_image in x_images:
            x_image = self.resnet3D.stem(x_image)

            x_image = self.resnet3D.layer1(x_image)
            x_image = self.resnet3D.layer2(x_image)
            x_image = self.resnet3D.layer3(x_image)
            x_image = self.resnet3D.layer4(x_image)

            x_image = self.resnet3D.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet3D.fc(x_image)
            deep_features.append(x_image)

        outputs = torch.cat(deep_features, dim=0)


        return outputs

class resnet_bilstm_transformer(nn.Module):
    def __init__(self,
                 num_classes,
                 num_frozen_layer=-1,
                 hidden_feature_size=128,
                 hidden_size_lstm=128,
                 bidirectional=False):
        super(resnet_bilstm_transformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_feature_size = hidden_feature_size
        self.hidden_size_lstm = hidden_size_lstm


        self.resnet = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(self.resnet, num_frozen_layer)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=self.hidden_feature_size)


        self.lstm = nn.LSTM(input_size=self.hidden_feature_size, hidden_size=self.hidden_size_lstm, bidirectional=bidirectional)


        self.pos_encoder = PositionalEncoding(d_model=self.hidden_size_lstm*2 if bidirectional else self.hidden_size_lstm, dropout=0.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size_lstm*2 if bidirectional else self.hidden_size_lstm, nhead=8, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)



        self.fc1 = nn.Linear(in_features=self.hidden_size_lstm*2 if bidirectional else self.hidden_size_lstm, out_features=self.num_classes)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x_images):

        deep_features = []
        for x_image in x_images:
            x_image = self.resnet.conv1(x_image)
            x_image = self.resnet.bn1(x_image)
            x_image = self.resnet.relu(x_image)
            x_image = self.resnet.maxpool(x_image)

            x_image = self.resnet.layer1(x_image)
            x_image = self.resnet.layer2(x_image)
            x_image = self.resnet.layer3(x_image)
            x_image = self.resnet.layer4(x_image)

            x_image = self.resnet.avgpool(x_image)
            x_image = torch.flatten(x_image, 1)
            x_image = self.resnet.fc(x_image)
            deep_features.append(x_image)

        deep_features_pack = pack_sequence(sequences=deep_features, enforce_sorted=False)

        outputs, (hn, cn) = self.lstm(deep_features_pack)
        # hn = hn.squeeze(dim=0)
        # outputs = self.fc1(hn)
        # outputs = self.softmax(outputs)


        '''
        为了测试经过pack_sequence之后LSTM的输出hn是否还是batch中原始顺序输出的
        事实证明是的，因此可直接在hn基础上进行预测分类，并能够和labels对应
        '''
        outputs_unpacked, lens_unpacked = pad_packed_sequence(outputs, batch_first=False, padding_value=0.0)
        s, n, e = outputs_unpacked.shape

        # '''添加标志位'''
        # cls = torch.zeros((1, n, e)).to(device)
        # cls[0, :, 0] = 1.0
        # outputs_unpacked = torch.cat((cls, outputs_unpacked), 0)
        # s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        # src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        # for i in range(lens_unpacked.shape[0]):
        #     src_key_padding_mask[i, lens_unpacked[i]+1:] = True

        s, n, e = outputs_unpacked.shape    # 更新outputs_unpacked的shape，因为新添加了一位cls标志位用于分类
        src_key_padding_mask = torch.where(torch.ones((n, s)) == 1, False, True).to(device)
        for i in range(lens_unpacked.shape[0]):
            src_key_padding_mask[i, lens_unpacked[i]:] = True

        outputs_unpacked = self.pos_encoder(outputs_unpacked)

        outputs_trans = self.transformer_encoder(src=outputs_unpacked, src_key_padding_mask=src_key_padding_mask)


        outputs = torch.zeros((n, e)).to(device)
        for i in range(n):
            temp_output = outputs_trans[:lens_unpacked[i], i, :]
            outputs[i, :] = torch.mean(temp_output, dim=0)
        outputs = self.fc1(outputs)
        outputs = self.softmax(outputs)

        # temp = outputs_unpacked[0,lens_unpacked[0]-1,:]
        # temp2 = hn[0,0,:]
        print(outputs)
        return outputs




def train_model(model, criterion, optimizer, scheduler, best_model_dir, metrics, writer:SummaryWriter, dataloaders, decision_level='Tumor', decision_strategy='mean', num_epochs=25, is_inception=False, if_crop=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = {'ACC': 0.0,
                    'AUC': 0.0,
                    'AUPRC': 0.0}



    softmax = nn.Softmax(dim=1)


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_metrics = {'ACC': 0.0,
                         'AUC': 0.0,
                         'AUPRC': 0.0}

        # Each epoch has a training and validation phase
        for phase in datasets_name:
            if phase == datasets_name[0]:
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            crop_size = 1
            if if_crop == 'CenterCrop':
                crop_size = 1
            elif if_crop == 'FiveCrop':
                crop_size = 5
            elif if_crop == 'TenCrop':
                crop_size = 10
            else:
                crop_size = 1
            batch_size = dataloaders[phase].batch_size * crop_size
            sample_size = dataset_sizes[phase] * crop_size
            epoch_pred_proba = torch.zeros(sample_size, num_classes,dtype=torch.float32, requires_grad=False).detach()
            epoch_labels = torch.zeros(sample_size,dtype=torch.float32, requires_grad=False).detach()
            epoch_file_path = []


            # Iterate over data.
            for i, (inputs, labels, file_path) in enumerate(dataloaders[phase]):

                # if if_crop:
                #     bs, ncrops, c, h, w = inputs.size()

                # for i in range(ncrops):
                #     show_slices(inputs[0,i,:,:,:].numpy())

                # if model_name == "3Dresnet":
                #
                #     inputs = inputs.unsqueeze(dim=2) if if_crop else inputs.unsqueeze(dim=1)

                # if epoch > 0 and phase == 'train':
                #     pass


                if model_name == "resnet_bilstm":
                    for idx, input in enumerate(inputs):
                        input = torch.unsqueeze(input, 1)
                        input = input.repeat(1, 3, 1, 1)
                        inputs[idx] = input.to(device)
                elif model_name == "resnet_transformer":
                    for idx, input in enumerate(inputs):
                        input = torch.unsqueeze(input, 1)
                        input = input.repeat(1, 3, 1, 1)
                        inputs[idx] = input.to(device)
                        # for i in range(input.shape[0]):
                        #     show_slices(input[i,:,:,:].numpy())
                elif model_name == "resnet3D":
                    for idx, input in enumerate(inputs):
                        # show_slices(input.numpy())
                        input = torch.unsqueeze(input, 0)
                        input = torch.unsqueeze(input, 0)
                        input = input.repeat(1, 3, 1, 1, 1)
                        # input = input.repeat(1, 3, 1, 1)
                        inputs[idx] = input.to(device)
                else:
                    inputs = inputs.to(device)

                labels = labels.to(device)

                for item in file_path:
                    epoch_file_path.append(item)

                epoch_labels[i*batch_size:i*batch_size+batch_size] = labels.detach().clone()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == datasets_name[0]):

                    # if model_name == 'resnet_bilstm':
                    #     outputs = model(inputs)
                    #     loss = criterion(outputs, labels)
                    #
                    # if model_name == 'resnet_transformer':
                    #     outputs = model(inputs)
                    #     loss = criterion(outputs, labels)

                    outputs,_ = model(inputs)
                    loss = criterion(outputs, labels)

                    # else:
                        # if if_crop:
                        #     temp = inputs.view(-1, c, h, w)
                        #     temp2 = inputs.view(-1, 1, c, h, w)
                        #     outputs_crops = model(inputs.view(-1, 1, c, h, w) if model_name == "3Dresnet" else inputs.view(-1, c, h, w))
                        #     outputs = outputs_crops.view(bs, ncrops, -1).mean(1)
                        #     loss = criterion(outputs, labels)
                        # else:
                        #     outputs = model(inputs)
                        #     loss = criterion(outputs, labels)

                    pred_proba = softmax(outputs).detach().clone()
                    epoch_pred_proba[i * batch_size:i * batch_size + batch_size, :] = pred_proba
                    # _, preds = torch.max(outputs, 1)
                    # _, preds = torch.max(predroba, dim=1)



                    # backward + optimize only if in training phase
                    if phase == datasets_name[0]:
                        loss.backward()
                        optimizer.step()
                '''
                想尝试伪标签方法的，先暂停了
                '''

                # for i, fp in enumerate(file_path):
                #
                #     file_name = fp.replace("\\","/").split("/")[-1].split("_")
                #     patient_id = file_name[0] + "_" + file_name[1]
                #     file_idx = int(file_name[2][:-3])
                #
                #     if patient_id in pseudo_labels[phase]:
                #         pseudo_labels[phase][patient_id][file_idx] = labels[i].detach().cpu().numpy().astype(np.int)
                #     else:
                #         temp_arr = np.zeros(30, dtype=np.int)
                #         pseudo_labels[phase][patient_id] = temp_arr
                #         pseudo_labels[phase][patient_id][file_idx] = labels[i].detach().cpu().numpy().astype(np.int)
                #
                #     if patient_id in pseudo_probas[phase]:
                #         pseudo_probas[phase][patient_id][file_idx] = pred_proba[i].detach().cpu().numpy()
                #     else:
                #         temp_arr = np.zeros(30, dtype=np.float)
                #         pseudo_probas[phase][patient_id] = temp_arr
                #         pseudo_probas[phase][patient_id][file_idx] = pred_proba[i].detach().cpu().numpy()



                # statistics
                # running_loss += loss.item() * len(inputs) if model_name == "resnet_bilstm" or  else loss.item() * inputs.size(0)
                running_loss += loss.item() * len(inputs)
                # break


            if phase == datasets_name[0]:
                scheduler.step()

            epoch_labels = epoch_labels.numpy().astype(np.int)
            epoch_pred_proba = epoch_pred_proba.numpy().astype(np.float)
            epoch_loss = running_loss / sample_size


            epoch_metrics, _, epoch_pred_label = make_decision(epoch_metrics=epoch_metrics,dataset_sizes=dataset_sizes,
                                                                  phase=phase,
                                                                  epoch_pred_proba=epoch_pred_proba, epoch_labels=epoch_labels,
                                                                  epoch_file_path=epoch_file_path,decision_level=decision_level,decision_strategy=decision_strategy)
            writer.add_scalars(main_tag="loss", tag_scalar_dict={phase: epoch_loss}, global_step=epoch)
            writer.add_scalars(main_tag="ACC", tag_scalar_dict={phase: epoch_metrics['ACC']}, global_step=epoch)
            writer.add_scalars(main_tag="AUC", tag_scalar_dict={phase: epoch_metrics['AUC']}, global_step=epoch)
            writer.add_scalars(main_tag="AUPRC", tag_scalar_dict={phase: epoch_metrics['AUPRC']}, global_step=epoch)


            # deep copy the model
            if phase == 'val' and epoch_metrics[metrics] > best_metrics[metrics]:
                print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f} AUPRC: {:.4f}    *'.format(
                    phase, epoch_loss, epoch_metrics['ACC'], epoch_metrics['AUC'], epoch_metrics['AUPRC']))
                best_metrics[metrics] = epoch_metrics[metrics]
                best_model_wts = copy.deepcopy(model.state_dict())
                # path = best_model_dir + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())+'.pt'
                path = best_model_dir + '/best_val_model.pt'
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss': epoch_loss
                            }, path)
            else:
                print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f} AUPRC: {:.4f}'.format(
                    phase, epoch_loss, epoch_metrics['ACC'], epoch_metrics['AUC'], epoch_metrics['AUPRC']))



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val {}: {:4f}'.format(metrics, best_metrics[metrics]))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, decision_level="Tumor", decision_strategy="mean", is_inception=False, if_crop=False):

    model.eval()  # Set model to evaluate mode
    phase = 'test'

    running_loss = 0.0
    running_corrects = 0
    fold_metrics = {'ACC': 0.0,
                     'AUC': 0.0,
                     'AUPRC': 0.0}

    softmax = nn.Softmax(dim=1)

    crop_size = 1
    if if_crop == 'CenterCrop':
        crop_size = 1
    elif if_crop == 'FiveCrop':
        crop_size = 5
    elif if_crop == 'TenCrop':
        crop_size = 10
    else:
        crop_size = 1
    batch_size = dataloaders[phase].batch_size * crop_size
    sample_size = dataset_sizes[phase] * crop_size
    fold_pred_proba = torch.zeros(sample_size, num_classes, dtype=torch.float32, requires_grad=False).detach()
    fold_hidden_state = torch.zeros(sample_size, model.fc1.in_features, dtype=torch.float32, requires_grad=False).detach()
    fold_labels = torch.zeros(sample_size, dtype=torch.float32, requires_grad=False).detach()
    fold_file_path = []



    for i, (inputs, labels, file_path) in enumerate(dataloaders[phase]):

        if model_name == "resnet_bilstm":
            for idx, input in enumerate(inputs):
                input = torch.unsqueeze(input, 1)
                input = input.repeat(1, 3, 1, 1)
                inputs[idx] = input.to(device)
        elif model_name == "resnet_transformer":
            for idx, input in enumerate(inputs):
                input = torch.unsqueeze(input, 1)
                input = input.repeat(1, 3, 1, 1)
                inputs[idx] = input.to(device)
                # for i in range(input.shape[0]):
                #     show_slices(input[i,:,:,:].numpy())
        elif model_name == "resnet3D":
            for idx, input in enumerate(inputs):
                # show_slices(input.numpy())
                input = torch.unsqueeze(input, 0)
                input = torch.unsqueeze(input, 0)
                input = input.repeat(1, 3, 1, 1, 1)
                # input = input.repeat(1, 3, 1, 1)
                inputs[idx] = input.to(device)
        else:
            inputs = inputs.to(device)

        labels = labels.to(device)

        for item in file_path:
            fold_file_path.append(item)

        fold_labels[i * batch_size:i * batch_size + batch_size] = labels.detach().clone()

        outputs, h_s = model(inputs)
        loss = criterion(outputs, labels)


        pred_proba = softmax(outputs).detach().clone()

        fold_pred_proba[i * batch_size:i * batch_size + batch_size, :] = pred_proba
        fold_hidden_state[i * batch_size:i * batch_size + batch_size, :] = h_s.detach().clone()



    fold_labels = fold_labels.numpy().astype(np.int)
    fold_pred_proba = fold_pred_proba.numpy().astype(np.float)
    fold_hidden_state = fold_hidden_state.numpy().astype(np.float)
    epoch_loss = running_loss / sample_size

    fold_metrics, _, fold_pred_label = make_decision(epoch_metrics=fold_metrics, dataset_sizes=dataset_sizes,
                                                          phase=phase,
                                                          epoch_pred_proba=fold_pred_proba, epoch_labels=fold_labels,
                                                          epoch_file_path=fold_file_path,
                                                          decision_level=decision_level,
                                                          decision_strategy=decision_strategy)
    print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f} AUPRC: {:.4f}'.format(
        phase, epoch_loss, fold_metrics['ACC'], fold_metrics['AUC'], fold_metrics['AUPRC']))
    return fold_metrics, fold_pred_proba, fold_pred_label, fold_labels, fold_file_path, fold_hidden_state


if __name__ == '__main__':


    torch.manual_seed(35)

    networks = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    batch_size = 8
    use_pretrained = True
    # decision_level = ['Tumor', 'Slices']
    num_epochs = 40
    decision_level = 'Tumor'
    decision_strategy = 'max'
    metrics = 'AUC'
    model_name = 'resnet_transformer'
    resnet_type = '50'
    dropout = 0.4
    num_workers = 0 if "windows" in platform.platform() else 4  # 如果是windows操作系统为0，否则为4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_folds = 10
    is_all_slices = True
    dataset_version = 'version15'
    num_slices = 1 if dataset_version == 'version4' else 3
    weight_decay = 0.001
    learning_rate = 0.0001
    num_frozen_layer = 340
    is_pT = True
    crop_type = [None, 'CenterCrop', 'FiveCrop', 'TenCrop']
    if_crop = crop_type[0]
    if_transform = if_crop
    class_weight = 681/np.array([60, 224, 220, 90, 45, 32, 10], dtype='float32') if is_pT else 681/np.array([583, 98], dtype='float32')
    class_weight = np.array([1, 1, 1, 1, 1, 1, 1], dtype='float32') if is_pT else np.array([1, 1], dtype='float32')
    interpolation_mode = InterpolationMode.NEAREST
    # input_size = 187
    num_classes = 7 if is_pT else 2

    '''
    Training phase
    '''
    #
    # for fold_idx in range(num_folds):
    #
    #
    #     model_conv, _ = initialize_model(model_name=model_name,
    #                                      num_classes=num_classes,
    #                                      num_frozen_layer=num_frozen_layer,
    #                                      use_pretrained=use_pretrained,
    #                                      num_slices=num_slices,
    #                                      resnet_type=resnet_type,
    #                                      dropout=dropout)
    #     '''
    #     实验数据读取
    #     '''
    #     datasets_name = ['train', 'val', 'test']
    #
    #     if if_crop != None:
    #         data_transforms = {
    #             datasets_name[0]: transforms.Compose([
    #                 # transforms.Resize(256, interpolation=interpolation_mode),
    #                 # transforms.RandomVerticalFlip(),
    #                 # transforms.TenCrop(input_size),
    #                 transforms.Lambda(_my_crops(if_crop, input_size))
    #                 # transforms.Lambda(_my_stack),    # windows下该方式会导致出错，仅在linux下可用 transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    #                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ]),
    #             datasets_name[1]: transforms.Compose([
    #                 # transforms.Resize(256, interpolation=interpolation_mode),
    #                 # transforms.RandomVerticalFlip(),
    #                 # transforms.TenCrop(input_size),
    #                 transforms.Lambda(_my_crops(if_crop, input_size))
    #                 # transforms.Lambda(_my_stack),    # windows下该方式会导致出错，仅在linux下可用 transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    #                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ]),
    #             datasets_name[2]: transforms.Compose([
    #                 # transforms.Resize(256, interpolation=interpolation_mode),
    #                 # transforms.RandomVerticalFlip(),
    #                 # transforms.TenCrop(input_size),
    #                 transforms.Lambda(_my_crops(if_crop, input_size))
    #                 # transforms.Lambda(_my_stack),    # windows下该方式会导致出错，仅在linux下可用 transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    #                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ]),
    #         }
    #
    #
    #
    #     writer = SummaryWriter(log_dir='logs/' + model_name + '_' + dataset_version + '/' + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + '/Fold' + str(fold_idx))
    #     data_dir = 'data/CT/' + dataset_version + '/Fold' + str(fold_idx)
    #     best_model_dir = 'model/' + model_name + '_' + dataset_version +'/Fold' + str(fold_idx)
    #     image_datasets = {x: MyDatasetFolder(root=os.path.join(data_dir, x),
    #                                          loader=img_tensor_loader,
    #                                          extensions=IMG_EXTENSIONS,
    #                                          transform=data_transforms[x] if if_crop != None else None
    #                                          )
    #                       for x in datasets_name}
    #
    #
    #     dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, collate_fn=MyCollate_fn if is_all_slices else None)
    #                    for x in datasets_name}
    #     dataset_sizes = {x: len(image_datasets[x]) for x in datasets_name}
    #     class_names = image_datasets[datasets_name[0]].classes
    #
    #
    #
    #
    #     model_conv = model_conv.to(device)
    #
    #
    #     weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
    #     criterion = nn.CrossEntropyLoss(weight=weight)
    #     # criterion = FocalLoss(gamma=5,alpha=weight)
    #
    #     # Observe that only parameters of final layer are being optimized as
    #     # opposed to before.
    #     params_to_update = model_conv.parameters()
    #     print("Params to learn:")
    #     if num_frozen_layer > 0:
    #         params_to_update = []
    #         for name, param in model_conv.named_parameters():
    #             if param.requires_grad == True:
    #                 params_to_update.append(param)
    #                 print("\t", name)
    #     else:
    #         for name, param in model_conv.named_parameters():
    #             if param.requires_grad == True:
    #                 print("\t", name)
    #
    #
    #     # optimizer_conv = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    #     optimizer_conv = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    #
    #     # Decay LR by a factor of 0.1 every 7 epochs
    #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.8)
    #     model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                              exp_lr_scheduler,
    #                              dataloaders=dataloaders,
    #                              best_model_dir=best_model_dir,
    #                              metrics=metrics, writer=writer,
    #                              num_epochs=num_epochs, decision_level=decision_level,
    #                              decision_strategy=decision_strategy,
    #                              is_inception=(model_name=="inception"),
    #                              if_crop=if_crop)
    # writer.flush()
    # writer.close()




    '''
    Test phase (10 folds cross validation strategy)
    '''
    # all_probas = []
    # all_labels = []
    # all_file_path = []
    # all_hidden_state = []
    # for fold_idx in range(num_folds):
    #     data_dir = 'data/CT/' + dataset_version + '/Fold' + str(fold_idx)
    #     best_model_dir = 'model/' + model_name + '_' + dataset_version + '/Fold' + str(fold_idx)
    #
    #     checkpiont = torch.load(best_model_dir + '/best_val_model.pt')
    #     model_conv, input_size = initialize_model(model_name=model_name,
    #                                              num_classes=num_classes,
    #                                              num_frozen_layer=num_frozen_layer,
    #                                              use_pretrained=use_pretrained,
    #                                              num_slices=num_slices,
    #                                              resnet_type=resnet_type,
    #                                              dropout=dropout)
    #     model_conv.load_state_dict(checkpiont["model_state_dict"])
    #
    #     model_conv = model_conv.to(device)
    #     if if_crop:
    #         data_transforms = {
    #             'test': transforms.Compose([
    #                 # transforms.Resize(256, interpolation=interpolation_mode),
    #                 # transforms.RandomVerticalFlip(),
    #                 # transforms.TenCrop(input_size),
    #                 transforms.Lambda(_my_crops(if_crop, input_size))
    #                 # transforms.Lambda(_my_stack),    # windows下该方式会导致出错，仅在linux下可用 transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    #                 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ]),
    #         }
    #
    #
    #     datasets_name = ['test']
    #     image_datasets = {x: MyDatasetFolder(root=os.path.join(data_dir, x),
    #                                          loader=img_tensor_loader,
    #                                          extensions=IMG_EXTENSIONS,
    #                                          transform=data_transforms[x] if if_crop != None else None
    #                                          )
    #                       for x in datasets_name}
    #
    #     dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers,
    #                                  drop_last=False, collate_fn=MyCollate_fn if is_all_slices else None)
    #                    for x in datasets_name}
    #     dataset_sizes = {x: len(image_datasets[x]) for x in datasets_name}
    #     class_names = image_datasets['test'].classes
    #
    #
    #     weight = torch.tensor(class_weight, dtype=torch.float32).to(device)
    #     criterion = nn.CrossEntropyLoss(weight=weight)
    #
    #     print("Performance on test set of Fold{}:".format(fold_idx))
    #     fold_metrics, fold_pred_proba, fold_pred_label, fold_labels, fold_file_path, fold_hidden_state = test_model(model_conv, decision_level=decision_level, decision_strategy=decision_strategy, if_crop=if_crop)
    #     all_probas.extend(fold_pred_proba.tolist())
    #     all_labels.extend(fold_labels.tolist())
    #     all_file_path.extend(fold_file_path)
    #     all_hidden_state.extend(fold_hidden_state.tolist())
    #
    # all_probas = np.array(all_probas)
    # all_labels = np.array(all_labels)
    # all_hidden_state = np.array(all_hidden_state)
    # all_AUC = roc_auc_score(y_true=all_labels, y_score=all_probas, multi_class='ovr') if is_pT else roc_auc_score(y_true=all_labels, y_score=all_probas[:,1])
    #
    # all_pred_labels = np.argmax(all_probas, axis=1)
    # all_AUPRC = f1_score(y_true=all_labels, y_pred=all_pred_labels, average='macro') if is_pT else average_precision_score(y_true=all_labels, y_score=all_probas[:,1])
    # all_confusion_matrix = confusion_matrix(y_true=all_labels, y_pred=all_pred_labels)
    # all_ACC = np.sum(all_labels == all_pred_labels) / len(all_labels)
    #
    # all_labels_onehot = np.eye(7)[np.array(all_labels)]
    # all_AUC_eachlabel = roc_auc_score(y_true=all_labels_onehot, y_score=all_probas, average=None) if is_pT else roc_auc_score(y_true=all_labels, y_score=all_probas[:,1])
    #
    #
    #
    # print('The final cross-validation results: Acc: {:.4f} AUC: {:.4f} AUPRC: {:.4f}'.format(
    #       all_ACC, all_AUC, all_AUPRC))
    # print(all_confusion_matrix)
    #
    #
    # dict_results = {
    #     'all_probas': all_probas,
    #     'all_labels': all_labels,
    #     'all_hidden_state': all_hidden_state,
    #     'all_file_path': all_file_path
    # }
    #
    # save_path = 'model/' + model_name + '_' + dataset_version + '/all_results.pkl'
    #
    # save_dict(dict_results, save_path)





    # '''
    # 导出隐藏特征
    # '''

    # load_path = 'model/' + model_name + '_' + dataset_version + '/all_results.pkl'
    # results_all = load_dict(load_path)
    # all_probas = results_all['all_probas']
    # all_labels = results_all['all_labels']
    # all_file_path = results_all['all_file_path']
    # all_hidden_state = results_all['all_hidden_state']

    # dict_hidden_state = {}

    # '''保存每个肿物的hidden_state到csv文件'''
    # for i, item in enumerate(all_file_path):
    #     file_name = item.replace("\\","/").split("/")   # 为了兼容windows和linux
    #     label = str(file_name[-2])
    #     label2 = str(all_labels[i])
    #     if label != label2:
    #         Exception('The labels of CT are not consistant !!! Be care for!!!')
    #     patient_ID = file_name[-1][:-3]
    #     ID_1, ID_2 = patient_ID.split("_")[0], patient_ID.split("_")[1]
    #     hidden_state = all_hidden_state[i]
    #     probs = all_probas[i]
    #     if patient_ID not in dict_hidden_state.keys():
    #         temp1 = hidden_state.tolist()
    #         temp3 = probs.tolist()
    #         temp2= [ID_1, ID_2, label, label2]
    #         temp2.extend(temp1)
    #         temp2.extend(temp3)
    #         dict_hidden_state[patient_ID] = temp2
    #     else:
    #         Exception('Duplicated hidden state for file {}'.format(patient_ID))

    # df_hidden_state = pd.DataFrame.from_dict(data=dict_hidden_state, orient='index')


    # df_save_path = 'model/' + model_name + '_' + dataset_version + '/hidden_state_pT_probs.csv'
    # df_hidden_state.to_csv(df_save_path,encoding='utf-8')


