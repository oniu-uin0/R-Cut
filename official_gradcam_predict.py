import argparse
import glob
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from collections import Counter
# from my_cam_model import vit_cam_patch16_224_in21k as create_model
from vit_model import vit_base_patch16_224_in21k as create_model
# from conformer import conformer_tscam_small_patch16 as create_model
# from my_feagradcam_model import feagradcam_big as create_model
# from swin_vit_model import swin_tiny_patch4_window7_224 as create_model
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
        default='./test_multiclass',
        # default="/home/y_niu/workspace6/work/dataset/imagenet/val/n07718747",
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
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
    weights_path = "./weights_imagenet/pre_base/model-40.pth"
    # weights_path = "./weights_no/jari_swin/model-100.pth"
    # weights_path = 'pre_weights/base_patch16_224_in21k.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location="cuda:1"))
    model.eval()
    # print(model)
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.blocks[i].norm1 for i in range(11, 12)]
    # target_layers = [model.conv_trans_12.trans_block.norm1]
    # target_layers = [model.layers[-1].blocks[-1].norm1]
    # target_layers = [model.layer4[-1]]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    # test_img = os.listdir(args.image_path)
    test_img = glob.glob(args.image_path+"/*.png")
    json_root = "test_last/label"
    in_list = []
    # for img_name in test_img:
        # img_path = os.path.join(args.image_path, img_name)
    img_name = 'test_multiclass/el2.png'
    rgb_img = cv2.imread(img_name, 1)[:, :, ::-1]
    im = rgb_img.copy()
    # im = cv2.resize(im, (224, 224))
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 340

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 16

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    mask_pred = grayscale_cam[0, :]
    mask_pred = cv2.resize(mask_pred, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
    # mask_pred = cv2.resize(mask_pred, (224, 224), interpolation=cv2.INTER_LINEAR)
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask_pred = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)
    # change the threshold of the  masked area
    _, mask_pred_binary_map = cv2.threshold(mask_pred,
                                            mask_pred.max() * 0.15, 1,
                                            cv2.THRESH_TOZERO)
    _, discard_map = cv2.threshold(mask_pred,
                                   mask_pred.max() * 0.1, 1,
                                   cv2.THRESH_TOZERO)
    # mask_image = (mask_pred_binary_map[..., np.newaxis] * im).astype("uint8")
    # plt.imshow(mask_image)
    # plt.imshow(mask_pred)
    # plt.imshow(mask_pred_binary_map)
    # plt.show()
    contours, _ = cv2.findContours((mask_pred_binary_map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # here maybe can find many minicontours
    if len(contours) != 0:
        # for contour in contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

#         # Determine whether the point is in the area
#         center_point = (int(x + w / 2), int(y + h / 2))
#         cls, point_pol = point_in_out(img_name, json_root, center_point)
#         if point_pol == 1.0 or point_pol == 0.0:
#             in_list.append(cls)
# in_dict = Counter(in_list)
# print(in_dict)

        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
    x1, y1, x2, y2 = estimated_bbox
    # im_box = cv2.rectangle(np.array(im), (x1, y1), (x2, y2), color1, 2)
    # im_box = cv2.applyColorMap(im_box, cv2.COLORMAP_RAINBOW)
    heatmap = cv2.applyColorMap(np.uint8(255 * discard_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_image = heatmap + np.float32(im) / 255
    cam_image = cam_image / np.max(cam_image)
    cam_image = np.uint8(cam_image * 255)
    cam_image = cv2.cvtColor(np.array(cam_image), cv2.COLOR_RGB2BGR)
    # im_box = cv2.rectangle(cam_image, (x1, y1), (x2, y2), color1, 10)
    # cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # cam_image = cv2.addWeighted(im_box, 0.8, heatmap[:, :, ::-1], 1, 0.8)
    plt.imshow(cam_image)
    # # plt.imshow(im_box)
    plt.show()
    # save_path = os.path.join("./outputs_imagenet", weights_path.split("/")[-2], 'score')
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    # cv2.imwrite(os.path.join(save_path, img_name.split("/")[-1]), cam_image[:, :, ::-1])
    # # plt.imshow(cam_image)
    # cv2.imshow("cam", cam_image)
    # # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
