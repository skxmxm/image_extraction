import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim
import cv2 as cv
from moviepy.editor import VideoFileClip, ImageSequenceClip
import shutil
from tqdm import tqdm
import ffmpeg

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

VIDEO_NAME = 'test_video.mp4'


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '_temp.png')
    img_mask = cv.imread(d_dir + imidx + '_temp.png', cv.IMREAD_GRAYSCALE)
    os.remove(d_dir + imidx + '_temp.png')
    img_list = image.tolist()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_list[i][j].reverse()
            if img_mask[i][j] != 0:
                img_list[i][j].append(img_mask[i][j])
            else:
                img_list[i][j][0] = 0
                img_list[i][j][1] = 0
                img_list[i][j][2] = 0
                img_list[i][j].append(0)
    png_img = np.array(img_list)
    cv.imwrite(d_dir + imidx + '_temp' + '.png', png_img)

    # cv.imwrite(d_dir+imidx+'result'+'.png', img)


def extraction():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp

    video_dir = os.path.join(os.getcwd(), 'test_data', 'videos_temp_data')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', 'videos_temp_data_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(video_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        # print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        # print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader), desc="视频处理中", total=len(img_name_list),
                                  leave=False):

        # print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


def video_dividing():
    video_path = os.path.join(os.getcwd(), 'test_data', 'input_videos', VIDEO_NAME)
    output_dir = os.path.join(os.getcwd(), 'test_data', 'videos_temp_data' + os.sep)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video = cv.VideoCapture(video_path)
    frame_num = 1
    if not video.isOpened():
        print("File not exist!")
    while True:
        flag, frame = video.read()
        if not flag:
            break
        cv.imwrite(output_dir + str(frame_num).zfill(6) + '.jpg', frame)
        frame_num += 1
    video.release()


def video_combination():
    partition_path = os.path.join(os.getcwd(), 'test_data', 'videos_temp_data_results')
    output_dir = os.path.join(os.getcwd(), 'test_data', 'videos_output' + os.sep)
    original_path = os.path.join(os.getcwd(), 'test_data', 'input_videos')
    original_video = VideoFileClip(os.path.join(original_path, VIDEO_NAME))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    name_list = VIDEO_NAME.split('.')[0:-1]
    new_name = ''
    for i in name_list:
        new_name += i + '.'
    new_name += 'avi'

    file_list = os.listdir(partition_path)
    # writer = cv.VideoWriter(os.path.join(output_dir, new_name), cv.VideoWriter_fourcc('D', 'I', 'V', 'X'),
    #                         original_video.fps, (original_video.w, original_video.h), True)
    #
    # for i in file_list:
    #     img = cv.imread(os.path.join(partition_path, i))
    #     writer.write(img)

    # writer.release()
    img_list = []
    for i in file_list:
        img = cv.imread(os.path.join(partition_path, i), cv.IMREAD_UNCHANGED)
        img_list.append(cv.cvtColor(img, cv.COLOR_BGRA2RGBA))

    clip = ImageSequenceClip(img_list, fps=original_video.fps, with_mask=True, ismask=False)
    audio = original_video.audio
    result_video = clip.set_audio(audio)
    result_video.write_videofile(os.path.join(output_dir, new_name), codec="png", fps=original_video.fps)


if __name__ == "__main__":
    # video_dividing()
    # extraction()
    video_combination()
    # shutil.rmtree(os.path.join(os.getcwd(), 'test_data', 'videos_temp_data' + os.sep))
    # shutil.rmtree(os.path.join(os.getcwd(), 'test_data', 'videos_temp_data_results' + os.sep))
