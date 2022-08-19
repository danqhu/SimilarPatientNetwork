import os
from pydicom import dcmread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import math
from sklearn.model_selection import StratifiedKFold
import copy

torch.manual_seed(23)

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        # self.ax.set_title('use scroll wheel to navigate images')
        self.ax.axis('off')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind, :, :], cmap=plt.cm.gray)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'down':
            if self.ind + 1 < self.slices:
                self.ind = self.ind + 1
        else:
            if self.ind-1 >= 0:
                self.ind = self.ind - 1
        self.update()

    def onscroll_loop(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        # plt.savefig('D:\Research\Cancer Multidisciplinary Team\Smart Staging\paper work\slice %s' % self.ind + 'image.png', bbox_inches='tight', pad_inches=0)

def show_slices(slices):
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    fig.tight_layout()
    tracker = IndexTracker(ax, slices)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

class CTPreprocessor(object):
    def __int__(self):
        self.dcm_files = None

    @classmethod
    def get_pixels(cls, slices):
        slices_ori = np.zeros((512, 512))
        for i, slice in enumerate(slices):
            if i == 0:
                slices_ori = np.array(slice.pixel_array.copy(), dtype=np.int32)[np.newaxis, :]
            else:
                temp = np.array(slice.pixel_array.copy(), dtype=np.int32)[np.newaxis, :]
                slices_ori = np.concatenate((slices_ori, temp), axis=0)
        return slices_ori

    @classmethod
    def get_pixels_HU(cls, slices):
        slices_HU = np.zeros((512, 512))
        for i, slice in enumerate(slices):

            slice_HU = np.array(slice.pixel_array.copy(), dtype=np.int32)
            intercept = slice.RescaleIntercept
            slope = slice.RescaleSlope
            slice_HU = slice_HU * slope + intercept
            if i == 0:
                slices_HU = np.array(slice_HU.copy(), dtype=np.float32)[np.newaxis, :]
            else:
                temp = np.array(slice_HU.copy(), dtype=np.float32)[np.newaxis, :]
                slices_HU = np.concatenate((slices_HU, temp), axis=0)

        return slices_HU

    @classmethod
    def set_window_level_and_size(cls, slices, HU_min, HU_max):
        slices_HU = slices
        if HU_min == None or HU_max == None:
            return slices_HU
        else:
            slices_HU[slices_HU < HU_min] = HU_min
            slices_HU[slices_HU > HU_max] = HU_max
            return slices_HU

    @classmethod
    def get_ROI(cls, slices, annotations, size_ori, fix_size=-1, num_slices=3):
        slices_ROI = copy.deepcopy(slices)
        s, height, width = slices_ROI.shape
        width_ori = size_ori[0]
        height_ori = size_ori[1]
        x_topleft, y_topleft, width_ROI = annotations[3], annotations[4], annotations[5]
        # 针对原始图像(512, 512)或者各项同性后图像(x, y)
        if width == width_ori and height == height_ori:
            pass
        else:
            x_topleft = width / width_ori * x_topleft
            y_topleft = height / height_ori * y_topleft
            width_ROI = width / width_ori * width_ROI

        idx_start = int(annotations[8]) - 1
        idx_end = int(annotations[7]) - 1
        num_annotated_slices = idx_end - idx_start + 1

        if fix_size < 0 or width_ROI >= fix_size:        # 如果设定的ROI区域大小小于0，则按标注大小截取
            if num_annotated_slices >= num_slices:        # 如果标注的层数大于设定的层数，则按标注层数截取
                slices_ROI = slices_ROI[idx_start:idx_end+1,
                                        int(y_topleft):int(y_topleft) + int(width_ROI),
                                        int(x_topleft):int(x_topleft) + int(width_ROI)]
            else:                                         # 如果标注的层数小于设定的层数，则按设定的层数截取（可能需要padding）
                idx_start -= math.floor((num_slices - num_annotated_slices)/2)
                idx_end += math.ceil((num_slices - num_annotated_slices)/2)
                front_padding = 0
                back_padding = 0
                if idx_start < 0:
                    front_padding = 0-idx_start
                if back_padding > s:
                    back_padding = idx_end - s

                slices_ROI = np.pad(slices_ROI, pad_width=((front_padding, back_padding), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0)
                idx_start += front_padding
                idx_end += back_padding
                slices_ROI = slices_ROI[idx_start:idx_end+1,
                             int(y_topleft):int(y_topleft) + int(width_ROI),
                             int(x_topleft):int(x_topleft) + int(width_ROI)]

        elif fix_size == 0:
            raise ValueError("The size of ROI can not be 0, please set a reasonable value!")
        else:
            x_topleft = int(x_topleft + width_ROI/2 - fix_size/2)
            y_topleft = int(y_topleft + width_ROI/2 - fix_size/2)
            width_ROI = fix_size

            left_padding = 0
            right_padding = 0
            top_padding = 0
            buttom_padding = 0
            front_padding = 0
            back_padding = 0

            if x_topleft < 0:
                left_padding = 0 - x_topleft
            if x_topleft + width_ROI > width:
                right_padding = x_topleft + width_ROI - width
            if y_topleft < 0:
                top_padding = 0 - y_topleft
            if y_topleft + width_ROI > height:
                buttom_padding = y_topleft + width_ROI - height

            if num_annotated_slices < num_slices:
                idx_start -= math.floor((num_slices - num_annotated_slices) / 2)   #padding数为单数时，在后面多padding一张
                idx_end += math.ceil((num_slices - num_annotated_slices) / 2)

                if idx_start < 0:
                    front_padding = 0 - idx_start
                if back_padding > s:
                    back_padding = idx_end - s

                slices_ROI = np.pad(slices_ROI, pad_width=((front_padding, back_padding), (0, 0), (0, 0)),
                                    mode='constant', constant_values=0)



            pad_width = np.max([left_padding, right_padding, top_padding,buttom_padding])
            slices_ROI = np.pad(slices_ROI,pad_width=((front_padding,back_padding),(pad_width,pad_width),(pad_width,pad_width)), mode='constant', constant_values=0)
            # show_slices(slices_ROI)

            x_topleft += pad_width
            y_topleft += pad_width
            idx_start += front_padding
            idx_end += back_padding


            slices_ROI = slices_ROI[idx_start:idx_end+1,
                         int(y_topleft) :int(y_topleft) + int(width_ROI),
                         int(x_topleft) :int(x_topleft) + int(width_ROI)]




        return slices_ROI, (x_topleft, y_topleft, width_ROI, idx_start, idx_end)

    @classmethod
    def normalize_slice(cls, slices, HU_min, HU_max):
        slice_normalized = slices.copy()

        slice_normalized = (slice_normalized-HU_min)/(HU_max-HU_min)*2.0-1.0

        return slice_normalized

    def standarize_slice(self, slices):

        return None

    @classmethod
    def get_isotropic_array(cls, slices, target_spacing=[0.7, 0.7, 5], is_ROI=False):

        sitk_slices = slices
        new_spaceing = np.array(target_spacing, dtype=np.float)
        spacing = np.array(sitk_slices.GetSpacing())

        if float('%.2f' % spacing[2]) != 5.0:
            print('The image thickness is not equal to 5mm!')
        size = np.array(sitk_slices.GetSize())           # 原始突变的大小，作为返回值一同返回，用于ROI区域的裁剪（需根据标注坐标和原始大小，确定各向同性后的图像ROI区域）
        direction = sitk_slices.GetDirection()
        origin = sitk_slices.GetOrigin()

        new_size = size * spacing / new_spaceing
        new_size = [int(s) for s in new_size]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(sitk_slices.GetDirection())
        resampler.SetOutputOrigin(sitk_slices.GetOrigin())
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spaceing)

        if is_ROI:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)


        slices_resampled = resampler.Execute(sitk_slices)

        print("The original slices' size is {}, and the spacing is {}".format(sitk_slices.GetSize(),
                                                                              sitk_slices.GetSpacing()))
        print("The resampled slices' size is {}, and the spacing is {}".format(slices_resampled.GetSize(),
                                                                               slices_resampled.GetSpacing()))

        return slices_resampled   # 注意：这里利用simpleITK的GetSize()方法得到的维度信息是[width, height, depth]， 而利用GetArrayFromImage()得到的数组维度为[depth, heigh, width]

    @classmethod
    def get_original_image(cls, folder_path):

        reader = sitk.ImageSeriesReader()
        dicom_file_names = reader.GetGDCMSeriesFileNames(folder_path)
        reader.SetFileNames(dicom_file_names)

        sitk_slices = reader.Execute()

        slices_HU = sitk.GetArrayFromImage(sitk_slices)  # 该方法得到的就是HU值

        return sitk_slices  # 注意：这里利用simpleITK的GetSize()方法得到的维度信息是[width, height, depth]， 而利用GetArrayFromImage()得到的数组维度为[depth, heigh, width]



'''
对照两个数据集中重复的影像数据(已核对)
'''
# CT1 = os.listdir('D:\Data\BUCH\CT1\matched')
# CT2 = os.listdir('D:\Data\BUCH\CT2\matched')
#
# for ct in CT1:
#     if ct in CT2:
#         print(ct)






'''
using pydicom to read .dcm files
'''
# folder_path = './CT/216449/839049/2/'
#
# slices_name = os.listdir(folder_path)
# slices_name.sort(key=len)   # make sure the slices listed in a correct rank I
# slices = []
# for f in slices_name:
#     slices.append(dcmread(folder_path + f))
# slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=False)    # make sure the slices listed in a correct rank II double check
#
# annotations = []
# annotations.append([216449, 839049, 3, 23, 27, (342, 323, 52)])
# spacing = [0.7, 0.7, 5]
#
#
# slices_ori = CTPreprocessor.get_pixels(slices)
# slices_HU = CTPreprocessor.get_pixels_HU(slices)
# slices_windowized = CTPreprocessor.set_window_level_and_size(slices_HU, -500, 1500)
# slices_normalized = CTPreprocessor.normalize_slice(slices_windowized)
# slices_ROI = CTPreprocessor.get_ROI(slices_normalized, annotations)
#
# # # using torchvision to crop and resize the images
# # slices_trans = torchvision.transforms.functional.center_crop(torch.tensor(slices_ROI), [64, 64])
# # slices_resized = torchvision.transforms.functional.resize(torch.tensor(slices_ROI), [224, 224])




'''
使用SimpleITK读取CT影像dicom文件，并进行各项同性转换、ROI区域裁剪、数据归一化等预处理操作
'''
def load_images_simpleITK(folder_path, annotation, fix_size=-1, num_slices=3, is_Resample=False):
    spacing = [0.7, 0.7, 5]
    HU_min = -1000
    HU_max = 400

    slices_ori = CTPreprocessor.get_original_image(folder_path)
    show_slices(sitk.GetArrayFromImage(slices_ori))

    if is_Resample:
        slices_resampled = CTPreprocessor.get_isotropic_array(slices_ori, target_spacing=spacing, is_ROI=False)
        # show_slices(sitk.GetArrayFromImage(slices_ori))
        # show_slices(sitk.GetArrayFromImage(slices_resampled))

    slices_windowized = CTPreprocessor.set_window_level_and_size(sitk.GetArrayFromImage(slices_resampled if is_Resample else slices_ori), HU_min, HU_max)
    slices_normalized = CTPreprocessor.normalize_slice(slices_windowized, HU_min, HU_max)
    slices_ROI, params = CTPreprocessor.get_ROI(slices_normalized, annotation, slices_ori.GetSize(), fix_size, num_slices=num_slices)
    # show_slices(slices_normalized)
    show_slices(slices_ROI)
    return slices_normalized, slices_ROI, params

def __save_file_in_the_folder(root, fold, datasets_split, dataset_name, fix_size=-1, is_all_slices=False, squence_type='consequent', num_slices=3, is_Resample=False, is_custom_data_augmentation=False, num_custom_aug=1000):
    CT1 = os.listdir('D:\Data\BUCH\CT1\matched')
    CT2 = os.listdir('D:\Data\BUCH\CT2\matched')

    folds_folder_path = os.path.join(root, fold)
    if not os.path.exists(folds_folder_path):
        os.makedirs(folds_folder_path)
    TVT_folder_path = os.path.join(folds_folder_path, dataset_name)
    if not os.path.exists(TVT_folder_path):
        os.makedirs(TVT_folder_path)

    for i, annotation in enumerate(datasets_split[fold][dataset_name]):
        label = datasets_split[fold][dataset_name+'_label'][i]

        pT_folder_path = os.path.join(TVT_folder_path, label)
        if not os.path.exists(pT_folder_path):
            os.makedirs(pT_folder_path)

        print('saving fold:{}, dataset:{}, file:{}, label:{}, No.:{}'.format(fold, dataset_name, str(annotation[1]) + '_' + str(annotation[2]), datasets_split[fold][dataset_name+'_label'][i], i+1))

        file_path = ""
        if str(annotation[0]) in CT1:
            file_path = "D:\Data\BUCH\CT1\matched" + "\\" + str(annotation[0]) + "\\" + str(
                annotation[1]) + "\\" + str(annotation[2])
        elif str(annotation[0]) in CT2:
            file_path = "D:\Data\BUCH\CT2\matched" + "\\" + str(annotation[0]) + "\\" + str(
                annotation[1]) + "\\" + str(annotation[2])
        else:
            raise ValueError(str(annotation[0]) + "_" + str(annotation[1]) + " do not find from the original folders")

        slices_normalized, slices_ROI, params = load_images_simpleITK(file_path, annotation, fix_size, num_slices=num_slices, is_Resample=is_Resample)   # fix_size用于是否读取指定大小的ROI区域，default值为-1表示根据标注截取
        show_slices(slices_normalized)
        show_slices(slices_ROI)
        file_save_folder_path = os.path.join(pT_folder_path, str(annotation[0]) + '_' + str(annotation[1]))

        if not is_all_slices:
            if squence_type == 'consequent':
                if slices_ROI.shape[0] < num_slices:   # 对于标注的影像没有要求的slices数目多时，需要padding缺失的层
                    num_padding = num_slices - slices_ROI.shape[0]
                    temp = np.pad(slices_ROI, pad_width=((0, num_padding), (0, 0), (0, 0)),
                                        mode='constant', constant_values=0)
                    # show_slices(temp)
                    save_path = file_save_folder_path + '_' + str(0) + '.pt'
                    if not os.path.isfile(save_path):
                        torch.save(torch.tensor(temp, dtype=torch.float32), save_path)
                else:
                    for sub_file_index in range(slices_ROI.shape[0] - num_slices + 1):      # 每num_slices连续的slice存储为一个.pt文件
                        save_path = file_save_folder_path + '_' + str(sub_file_index) + '.pt'
                        if not os.path.isfile(save_path):
                            temp = slices_ROI[sub_file_index:sub_file_index + num_slices, :, :]
                            torch.save(torch.tensor(temp, dtype=torch.float32), save_path)
            elif squence_type == 'identical':
                for sub_file_index in range(slices_ROI.shape[0]):      # 每一张slice重复成三个通道存储为一个.pt文件
                    save_path = file_save_folder_path + '_' + str(sub_file_index) + '.pt'
                    if not os.path.isfile(save_path):
                        temp = np.repeat(slices_ROI[sub_file_index, :, :][np.newaxis,:,:], repeats=num_slices, axis=0)
                        # show_slices(temp)
                        torch.save(torch.tensor(temp, dtype=torch.float32), save_path)
        else:
            save_path = file_save_folder_path + '.pt'
            if not os.path.isfile(save_path):
                torch.save(torch.tensor(slices_ROI, dtype=torch.float32), save_path)


        if is_custom_data_augmentation and dataset_name != 'test':

            augment_folder_path = folds_folder_path + '/custom_data_augmentation'
            if not os.path.exists(augment_folder_path):
                os.makedirs(augment_folder_path)

            roi = slices_ROI
            s, h, w = slices_normalized.shape
            x_topleft, y_topleft, width_ROI, idx_start, idx_end = params
            mold = np.arange(s * h * w).reshape(s, h, w)
            roi_index = mold[idx_start:idx_end + 1,
                        int(y_topleft):int(y_topleft) + int(width_ROI),
                        int(x_topleft):int(x_topleft) + int(width_ROI)]



            count_custom_aug = 0

            while count_custom_aug < num_custom_aug:

                x_rand = np.random.randint(low=0, high=w-width_ROI, size=1)[0]
                y_rand = np.random.randint(low=80, high=h-width_ROI-20, size=1)[0]
                start_rand = np.random.randint(low=0, high=s-num_slices, size=1)[0]
                print(str(x_rand), str(y_rand), str(start_rand))

                augment_index = mold[start_rand:start_rand + num_slices,
                                     int(y_rand):int(y_rand) + int(width_ROI),
                                     int(x_rand):int(x_rand) + int(width_ROI)]


                if np.intersect1d(roi_index, augment_index).shape[0] > 0:
                    continue
                else:
                    roi2 = slices_normalized[start_rand:start_rand + num_slices,
                                             int(y_rand):int(y_rand) + int(width_ROI),
                                             int(x_rand):int(x_rand) + int(width_ROI)]

                    save_path = augment_folder_path + '/' + str(annotation[1]) + "_" + str(annotation[2]) + '_aug_' + str(count_custom_aug) + '.pt'
                    if not os.path.isfile(save_path):
                        torch.save(torch.tensor(roi2, dtype=torch.float32), save_path)
                    # show_slices(roi2)
                    count_custom_aug += 1

def save_image_as_pt(root, datasets_split, datasets_name, fix_size, is_all_slices=False, squence_type='consequent', num_slices=3,is_Resample=False, is_custom_data_augmentation=False, num_custom_aug=1000):

    for fold in datasets_split.keys():
        for dn in datasets_name:
            __save_file_in_the_folder(root, fold, datasets_split, dn, fix_size=fix_size,is_all_slices=is_all_slices, squence_type=squence_type, num_slices=num_slices, is_Resample=is_Resample, is_custom_data_augmentation=is_custom_data_augmentation, num_custom_aug=num_custom_aug)




if __name__ == '__main__':

    random_seed = 23
    num_folds = 10
    val_set_size = 0.2
    root = 'data/CT/version17'
    datasets_name = ('train', 'val', 'test')
    datasets_split = {}
    squence_type = 'consequent'
    fix_size = 224
    num_slices = 3
    is_pT = False
    is_Resample = True
    is_N01vsN23 = True
    is_all_slices = True
    is_1234 = False
    is_custom_data_augmentation = False
    num_custom_aug = 100


    # data_image_df = pd.read_csv("image_annotation.csv")           # 读取标注数据 for old version N stage
    data_image_df = pd.read_csv("report_annotationV9_reviewedv2.0.csv")


    data_image_df.dropna(subset=["Stage_pT1234_8th", "Stage_pN", "主肿物位置_reviewed", "Enhanced"], inplace=True)  # 剔除出没有病理分期和增强CT影像的患者

    for x in data_image_df.index:
        if data_image_df.loc[x, "Stage_pN"] == "x":  # 提出病理分期为x的项
            data_image_df.drop(x, inplace=True)

    annotation_df = data_image_df[['Image_id',
                                   'CT_exam_id2',
                                   'Enhanced',
                                   'Top_Left_X.1',
                                   'Top_Left_Y.1',
                                   'Width.1',
                                   'Total_Slices_Num.1',
                                   'Index_Start_Slice.1',
                                   'Index_End_Slice.1']].astype("int")  # 读取每个患者影像的标注信息，用于ROI区域截取

    annotation = annotation_df.values

    labels_pT = data_image_df["Stage_pT1234_8th"].astype("string")
    labels_pT = copy.deepcopy(labels_pT)
    if is_1234:
        labels_pT[labels_pT == '1a'] = '1'
        labels_pT[labels_pT == '1b'] = '1'
        labels_pT[labels_pT == '1c'] = '1'
        labels_pT[labels_pT == '2a'] = '2'
        labels_pT[labels_pT == '2b'] = '2'


    print('pT stage:\n{}'.format(labels_pT.value_counts()))

    labels_pT = labels_pT.values

    labels_pN = data_image_df["Stage_pN"].astype("int32")
    labels_pN = copy.deepcopy(labels_pN)

    labels_pN[labels_pN == 0] = 'Neg'
    labels_pN[labels_pN == 1] = 'Neg' if is_N01vsN23 else 'Pos'
    labels_pN[labels_pN == 2] = 'Pos'
    labels_pN[labels_pN == 3] = 'Pos'


    print('pN stage:\n{}'.format(labels_pN.value_counts()))

    labels_pN = labels_pN.values


    if is_pT:
        label = labels_pT
    else:
        label = labels_pN






    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    skf2 = StratifiedKFold(n_splits=int(1/val_set_size), shuffle=True, random_state=random_seed)
    for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(label)), label)):
        labels_train_val = label[train_val_index]
        annotation_train_val = annotation[train_val_index, :]
        labels_test = label[test_index]
        annotation_test = annotation[test_index, :]

        for train_index, val_index in skf2.split(np.zeros(len(labels_train_val)), labels_train_val):
            labels_train = labels_train_val[train_index]
            annotation_train = annotation_train_val[train_index, :]

            labels_val = labels_train_val[val_index]
            annotation_val = annotation_train_val[val_index, :]

            break       # 目前只选取第一次train和val数据集划分

        datasets_split['Fold'+str(i)] = {'train_label': labels_train,
                                         'train': annotation_train,
                                         'val_label': labels_val,
                                         'val': annotation_val,
                                         'test_label': labels_test,
                                         'test': annotation_test}

    save_image_as_pt(root, datasets_split, datasets_name, fix_size=fix_size, is_all_slices=is_all_slices, squence_type=squence_type,
                     num_slices=num_slices, is_Resample=is_Resample, is_custom_data_augmentation=is_custom_data_augmentation,
                     num_custom_aug=num_custom_aug)




























    # if str(annotation[i][1]) == "1559991":
    #     flag = True
    #
    # if flag:
    #     slices_normalized, slices_ROI = load_images_simpleITK(file_path, annotation[i,:])
    #
    #     show_slices(slices_normalized)
    #
    #     show_slices(slices_ROI)
    #     # np.save('data/test' + str(annotation[i][2]), slices_ROI)
    #     # torch.save(torch.tensor(slices_ROI, dtype=torch.float32), 'data/test' + str(annotation[i][1])+'.pt')





















