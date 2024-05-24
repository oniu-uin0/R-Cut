import argparse
import glob
import json
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import os
from tools import xml_manage
from tools.metrics_test import *
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import Counter
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, \
    ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, \
    ROADLeastRelevantFirst, \
    ROADCombined, \
    ROADMostRelevantFirstAverage, \
    ROADLeastRelevantFirstAverage

from vit_model import vit_base_patch16_224_in21k as create_model

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/data/imagenet/img_val',
        # default='/data/CUB_200_2011/CUB_200_2011/images/006.Least_Auklet',
        # default='scorecut_test/test_contrast',
        # default="/home/y_niu/workspace6/work/dataset/imagenet/val/n07718747",
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        default=False,
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigengradcam')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def progress_mask(mask_pred):
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)

    return mask


def mask_cam_on_pic(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = progress_mask(heatmap)
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_RAINBOW)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(im) / 255
    cam_image = cam_image / np.max(cam_image)
    cam_image = np.uint8(cam_image * 255)
    cam_image = cv2.cvtColor(np.array(cam_image), cv2.COLOR_RGB2BGR)

    return cam_image


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :]
    # result = F.normalize(result, p=2)
    result = result.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swin_reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def point_in_out(img, json_pth, point):
    json_path = os.path.join(json_pth, img.split('/')[-1].split('.')[0] + ".json")
    with open(json_path, "r") as annotation:
        anno = json.load(annotation)

    bbox = anno['shapes'][0]['points']
    gt_cls = anno['shapes'][0]['label'].split('_')[0]
    bbox = np.array(bbox)
    bbox = np.array([[int(bbox[0][0]), int(bbox[0][1])],
                     [int(bbox[1][0]), int(bbox[0][1])],
                     [int(bbox[1][0]), int(bbox[1][1])],
                     [int(bbox[0][0]), int(bbox[1][1])]])[:, None, :]
    result = cv2.pointPolygonTest(bbox, point, False)
    return gt_cls, result


if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # create model
    model = create_model(num_classes=1000)
    # load model weights
    weights_path = "./weights_imagenet/model_adamw-95.pth"

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cuda:0"))
    model.eval()

    point_game_test = True
    bbox_iou_test = True
    save_img = True

    # print(model)
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[i].norm1 for i in range(11, 12)]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    # test_img = os.listdir(args.image_path)
    '''load label and xml'''
    with open('train.json') as label_json:
        label_dict = json.load(label_json)
    label_dict = {v: k for k, v in label_dict.items()}
    xmls_path = "/data/imagenet/bbox_val/val"
    """load img"""
    test_img = glob.glob(args.image_path + "/*.JPEG")
    # test_img = glob.glob(args.image_path + "/*.jpg")
    # test_img = glob.glob(args.image_path + "/*.png")
    imgs = sorted(test_img, key=lambda x: x.split("_")[-1][:-4])

    '''test cub'''
    # test_path = '/data/CUB_200_2011/CUB_200_2011/images'
    # cub_images_path = '/data/CUB_200_2011/CUB_200_2011/images.txt'
    # with open(cub_images_path) as cub_images:
    #     imgs = cub_images.readlines()
    # '''load cub label and xml'''
    # cub_label_path = '/data/CUB_200_2011/CUB_200_2011/image_class_labels.txt'
    # with open(cub_label_path) as cub_class_id:
    #     cub_class_id = cub_class_id.readlines()
    # cub_box_path = '/data/CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
    # with open(cub_box_path) as cub_boxes:
    #     cub_boxes = cub_boxes.readlines()

    basecam_iou = 0.0
    ncut_iou = 0.0
    scut_iou = 0.0

    basecam_index = 0
    ncut_index = 0
    scut_index = 0

    basecam_in = 0
    ncut_in = 0
    scut_in = 0

    '''bad data'''
    bad_test_number = 0
    bad_basecam_iou = 0.0
    bad_ncut_iou = 0.0
    bad_scut_iou = 0.0

    bad_basecam_index = 0
    bad_ncut_index = 0
    bad_scut_index = 0

    bad_basecam_in = 0
    bad_ncut_in = 0
    bad_scut_in = 0

    class_confidences = 0.0
    bad_class_confidences = 0.0
    cam_score_diff = 0.0
    cut_score_diff = 0.0
    scut_score_diff = 0.0

    test_number = 2000
    start = time.time()
    # for image_name in test_img:
    for i in range(test_number):
        '''read imagenet'''
        image_name = imgs[i]
        img_id = image_name.split('/')[-1][:-4]
        '''get the label information'''
        xml_pth = os.path.join(xmls_path, img_id + "xml")
        annotation = xml_manage.get_annotation(xml_pth)
        xy_bbox = np.array(annotation['xy_bbox'])
        wh_bbox = np.array(annotation["wh_bbox"])
        class_name = annotation['class_name']
        class_id = int(label_dict[class_name])
        '''read cub information'''
        # image_name = imgs[i]
        # image_name = image_name.split(' ')[-1].strip()
        # image_name = os.path.join(test_path, image_name)
        #
        # cub_box = cub_boxes[i].split(' ')[-4:]
        # xy_bbox = [int(float(a.strip())) for a in cub_box]
        # x1, y1, x2, y2 = xy_bbox
        # wh_bbox = [xy_bbox[0], xy_bbox[1], xy_bbox[2] - xy_bbox[0], xy_bbox[3] - xy_bbox[1]]
        # xy_bbox = np.array([[xy_bbox[0], xy_bbox[1]],
        #                     [xy_bbox[2], xy_bbox[1]],
        #                     [xy_bbox[2], xy_bbox[3]],
        #                     [xy_bbox[0], xy_bbox[3]]])
        #
        # class_id = int(float(cub_class_id[i].split(' ')[-1].strip()))
        """single_class"""
        # image_name = 'scorecut_test/test_imagenet/ILSVRC2012_val_00004345.JPEG'
        """multi_class
            161dog,bird87
            101 el, 340 zebra
            cat 282, dog 243,161"""
        # class_id = 340
        # image_name = 'scorecut_test/test_multiclass/catdog.png'
        # image_name = 'scorecut_test/test_multiclass/el2.png'
        # image_name = 'scorecut_test/test_multiclass/el4.png'

        # image_name = 'scorecut_test/test_bird/img_2.png'
        # image_name = '../TokenCut/examples/test.jpg'
        rgb_img = cv2.imread(image_name, 1)[:, :, ::-1]
        im = rgb_img.copy()
        # im = cv2.resize(im, (224, 224))
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        # input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        # target_category = 161
        '''161dog,bird87
            101 el, 340 zebra
            cat 282, dog 243,161
            '''
        targets = [ClassifierOutputSoftmaxTarget(class_id)]
        # targets = [ClassifierOutputTarget(class_id)]
        # targets = None
        # targets = [ClassifierOutputTarget(161)]
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 48

        grayscale_cams, save_map, class_confidence, pre_class = cam(input_tensor=input_tensor,
                                                                    targets=targets,
                                                                    eigen_smooth=args.eigen_smooth,
                                                                    aug_smooth=args.aug_smooth)

        # print(pre_class)
        # '''Here grayscale_cam has only one image in the batch'''
        # if class_confidence < 0.1 and pre_class != class_id:
        if pre_class != class_id:
            bad_test_number += 1
            save_img = True
            bad_class_confidences += class_confidence
            bad_data = True
        else:
            bad_data = False
            save_img = False
            class_confidences += class_confidence
        mask_pred = grayscale_cams[0, :]
        ncut_pre = save_map['ncut'][0, :]
        original_pred = save_map["base_cam"][0, :]
        # plt.imshow(ncut_pre)
        # plt.imshow(mask_pred)
        # plt.show()


        mask_pred = cv2.resize(mask_pred, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        original_pred = cv2.resize(original_pred, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        ncut_pred = cv2.resize(ncut_pre, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_pred = progress_mask(mask_pred)
        original_pred = progress_mask(original_pred)
        ncut_pred = progress_mask(ncut_pred)

        if bbox_iou_test:
            threshold = 0.2
            # change the threshold of the masked area
            _, mask_pred_binary_map = cv2.threshold(mask_pred,
                                                    mask_pred.max() * threshold, 1,
                                                    cv2.THRESH_TOZERO)

            _, original_pred_binary_map = cv2.threshold(original_pred,
                                                        original_pred.max() * threshold, 1,
                                                        cv2.THRESH_TOZERO)
            _, ncut_pred_binary_map = cv2.threshold(ncut_pred,
                                                    ncut_pred.max() * threshold, 1,
                                                    cv2.THRESH_TOZERO)
            mask_image = (mask_pred_binary_map[..., np.newaxis] * im).astype("uint8")
            # plt.imshow(grayscale_cams[0, :])
            # plt.imshow(mask_pred)
            # plt.imshow(mask_pred_binary_map)
            # plt.show()
            '''compute the point game'''
            if bad_data:
                bad_basecam_iou += jug_box_iou(original_pred_binary_map, xy_bbox)[0]
                bad_basecam_index += jug_box_iou(original_pred_binary_map, xy_bbox)[1]
                bad_scut_iou += jug_box_iou(mask_pred_binary_map, xy_bbox)[0]
                bad_scut_index += jug_box_iou(mask_pred_binary_map, xy_bbox)[1]
                scut_box = jug_box_iou(mask_pred_binary_map, xy_bbox)[2]
                bad_ncut_iou += jug_box_iou(ncut_pred_binary_map, xy_bbox)[0]
                bad_ncut_index += jug_box_iou(ncut_pred_binary_map, xy_bbox)[1]
                ncut_box = jug_box_iou(ncut_pred_binary_map, xy_bbox)[2]
            else:
                basecam_iou += jug_box_iou(original_pred_binary_map, xy_bbox)[0]
                basecam_index += jug_box_iou(original_pred_binary_map, xy_bbox)[1]
                scut_iou += jug_box_iou(mask_pred_binary_map, xy_bbox)[0]
                scut_index += jug_box_iou(mask_pred_binary_map, xy_bbox)[1]
                scut_box = jug_box_iou(mask_pred_binary_map, xy_bbox)[2]
                ncut_iou += jug_box_iou(ncut_pred_binary_map, xy_bbox)[0]
                ncut_index += jug_box_iou(ncut_pred_binary_map, xy_bbox)[1]
                ncut_box = jug_box_iou(ncut_pred_binary_map, xy_bbox)[2]

        x1, y1, x2, y2 = scut_box
        out_color = (0, 255, 0)
        gt_color = (255, 0, 0)

        cut_image = mask_cam_on_pic(mask_pred_binary_map)
        cam_image = mask_cam_on_pic(original_pred_binary_map)
        ncut_image = mask_cam_on_pic(ncut_pred_binary_map)
        # generate the box information
        # scut_image_box = cv2.rectangle(np.array(cut_image), (x1, y1), (x2, y2), out_color, 2)
        # scut_box_image = cv2.rectangle(np.array(scut_image_box), tuple(xy_bbox[0][0]), tuple(xy_bbox[0][2]), gt_color,
        #                                2)
        # put confidence on it
        # scut_box_image = cv2.putText(scut_box_image, 'Confidence: {:.4f}'.format(class_confidence), (20, 20),
        #                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # plt.imshow(scut_box_image)
        plt.imshow(cut_image)
        plt.show()

        # '''save result'''
        # if save_img:
        #     save_path = os.path.join("./outputs", image_name.split("/")[-2], "cut_{}".format(args.method))
        #     box_path = os.path.join(save_path, "box_image")
        #     if os.path.exists(box_path) is False:
        #         os.makedirs(box_path)
        #     cv2.imwrite(os.path.join(box_path, image_name.split("/")[-1]), scut_box_image[:, :, ::-1])

