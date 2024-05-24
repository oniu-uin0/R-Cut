import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import copy as cp
import random
import cv2
import collections
import os


def Vid2Frm():
    root_path = '/home/niu/下载/dataset/our'
    # seqs = os.listdir(os.path.join(os.getcwd(), 'videos'))
    seqs = os.listdir(os.path.join(root_path, 'nedo20191'))  # please take care of your own path

    for seq in seqs:
        # seq_path = os.path.join(os.getcwd(), 'videos', seq)
        # save_path = os.path.join(os.getcwd(), 'Frames', seq[:-4])
        seq_path = os.path.join(root_path, 'nedo20191', seq)
        save_path = os.path.join(root_path, 'Frames')
        # save_path = os.path.join(root_path, 'Frames', seq[:-4])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cap = cv2.VideoCapture(seq_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        frames_num = int(cap.get(7))
        counts = 0
        print(frames_num)
        for i in range(frames_num):
            ret, frame = cap.read()
            if frame is None:
                break
            if (i + 1) % 5 == 0:
                cv2.imwrite(os.path.join(save_path, str(seq[:-4]) + '_' + format(str(counts), '0>5s') + '.png'), frame)
                counts += 1
            # if not ret:
            #     print('video is all read')
            # break
        print("{} frames are extracted.".format(counts))
        cap.release()


def Frm2Vid():
    data_path = "./video1"
    frames = os.listdir(os.path.join(data_path))
    frames.sort(key=lambda x: int(x.split('_')[-1][:-4]))
    fps = 3  # 视频帧率
    size = (960, 720)  # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter("cam1.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    for f in frames:
        image_path = os.path.join(data_path, f)
        print(image_path)
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()


def divide_method2(img, m, n):  # 分割成m行n列
    h, w = img.shape[0], img.shape[1]
    grid_h = int(h * 1.0 / (m - 1) + 0.5)  # 每个网格的高
    grid_w = int(w * 1.0 / (n - 1) + 0.5)  # 每个网格的宽

    # 满足整除关系时的高、宽
    h = grid_h * (m - 1)
    w = grid_w * (n - 1)

    # 图像缩放
    img_re = cv2.resize(img, (w, h),
                        cv2.INTER_LINEAR)  # 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    # plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
    gx = gx.astype(np.int)
    gy = gy.astype(np.int)

    divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                            np.uint8)  # 这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息

    for i in range(m - 1):
        for j in range(n - 1):
            divide_image[i, j, ...] = img_re[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
    return divide_image


def display_blocks(divide_image):
    m, n = divide_image.shape[0], divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m, n, i * n + j + 1)
            plt.imshow(divide_image[i, j, :])
            # save_path = os.path.join("outputs_use", img_path.split(".")[0].split("/")[1])
            # if os.path.exists(save_path) is False:
            #     os.makedirs(save_path)
            # patch = divide_image[i, j, ...]
            # patch = patch[:, :, ::-1]
            # cv2.imwrite(os.path.join(save_path, str(i) + str(j) + '.png'), patch)
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Vid2Frm()
    # Frm2Vid()
    img_path = 'test_use/m3042ex1_front_03688.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(img_path.split('/')[1], img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[0], img.shape[1]
    # fig1 = plt.figure('原始图像')
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title('Original image')

    m = 4
    n = 4
    divide_image2 = divide_method2(img, m + 1, n + 1)  # 该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
    fig3 = plt.figure('分块后的子图像:图像缩放法', figsize=(m * 1.5, n * 1.5))
    display_blocks(divide_image2)
