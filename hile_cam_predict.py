import glob
import json
from collections import Counter

import scipy
from PIL import Image
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from vit_model import vit_base_patch16_224_in21k as create_model
from tools import xml_manage

# from my_cam_model import vit_cam_patch16_224_in21k as create_model
# from conformer import conformer_tscam_small_patch16 as create_model
# from my_feagradcam_model import feagradcam_big as create_model

CLS2IDX = {0: "cat",
           1: "dog"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def jug_point_game(mask_pred_binary_map, xy_bbox):
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
        box2 = [x, y, w, h]
        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
        """determine the IOU of the two boxes"""
        # if xml_manage.test_iou(wh_bbox, box2) > 0:
        #
        #     xml_manage.test_iou(wh_bbox, box2)
        """Determine whether the point is in the area"""
        center_point = (int(x + w / 2), int(y + h / 2))
        result = cv2.pointPolygonTest(xy_bbox, center_point, False)
        if result == 1.0 or result == 0.0:
            return 1
        else:
            return 0


def jug_box_iou(map, bbox):
    contours, _ = cv2.findContours((map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # here maybe can find many minicontours
    if len(contours) != 0:
        # for contour in contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        box2 = [x, y, w, h]
        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
        """determine the IOU of the two boxes"""
        if xml_manage.test_iou(bbox, box2) > 0:
            return xml_manage.test_iou(bbox, box2), 1
        else:
            return 0, 0


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    grads = []
    head_grads = []
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)  # (1,12,197,197)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    # grid_show(grads, 4)
    # grid_show(head_grads, 12)
    return R[0, 1:]


def grid_show(to_shows, cols):
    rows = (len(to_shows) - 1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            try:
                image = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            # axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.tight_layout()
    plt.show()


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    _, mask = cv2.threshold(mask,
                            mask.max() * 0.01, 1,
                            cv2.THRESH_TOZERO)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def progress_mask(mask_pred):
    mask_min_v, mask_max_v = mask_pred.min(), mask_pred.max()
    mask = (mask_pred - mask_min_v) / (mask_max_v - mask_min_v)
    return mask


def generate_visualization(input, im, class_index=None):
    transformer_attribution = generate_relevance(model, input.unsqueeze(0).cuda(), index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    # transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    # transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    # transformer_attribution = cv2.resize(transformer_attribution[0][0], (im.size[0], im.size[1]),
    #                                      interpolation=cv2.INTER_LINEAR)
    transformer_attribution = transformer_attribution[0][0].detach().cpu().numpy()
    featuremap = transformer_attribution
    transformer_attribution = cv2.resize(transformer_attribution,
                                         (im.size[0], im.size[1]), interpolation=cv2.INTER_LINEAR)
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / \
                              (transformer_attribution.max() - transformer_attribution.min())
    # image_transformer_attribution = input.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = np.array(im)
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    _, mask_pred_binary_map = cv2.threshold(transformer_attribution,
                                            transformer_attribution.max() * 0.01, 1,
                                            cv2.THRESH_TOZERO)
    contours, _ = cv2.findContours((mask_pred_binary_map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # here maybe can find many minicontours
    # if len(contours) != 0:
    #     # for contour in contours:
    #     c = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(c)
    #
    #     # Determine whether the point is in the area
    #     center_point = (int(x + w / 2), int(y + h / 2))
    #     cls, point_pol = point_in_out(img_name, json_root, center_point)
    #     if point_pol == 1.0 or point_pol == 0.0:
    #         return cls
    #         # in_list.append(cls)
    #     else:
    #         return None
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis, featuremap


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# initialize ViT pretrained
# create model
model = create_model(num_classes=200, has_logits=False).cuda()
# load model weights
# weights_path = "./jari_weights/model-9.pth"
# weights_path = "./weights/jari_vit/model-100.pth"
# weights_path = "cub_weights/model-50.pth"
# weights_path = "weights_clean/jari_vit/model-30.pth"
# weights_path = "./weights_imagenet/base_vit/model_adamw-30.pth"
# weights_path = "./weights_imagenet/freeze_dino_b/model_adamw-95.pth"
# weights_path = "dogcat_weights_vitbase/model-2.pth"
# weights_path = "cub_weights_conformer/model-29.pth"
# cub weights
weights_path = 'weights_cub/freeze_dino_b/best_model.pth'
# weights_path = 'weights_cub/base_vit_pre/model-50.pth'

save_result = False
point_game_test = True
bbox_iou_test = True

assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location="cuda:0"))
model.eval()

# test_path = './test_last'
# test_path = '/data/imagenet/test'
# test_pth = '/data/CUB_200_2011/CUB_200_2011/images/006.Least_Auklet'
# test_pth = "./scorecut_test/test_contrast"
# test_img = os.listdir(test_path)

'''test imagenet images'''
test_path = '/data/imagenet/img_val'
test_img = glob.glob(test_path + "/*.JPEG")
# test_img = glob.glob(test_pth+ "/*.jpg")
# test_img = glob.glob(test_pth + "/*.png")
imgs = sorted(test_img, key=lambda x: x.split("_")[-1][:-4])
'''load imagenet label and xml'''
with open('train.json') as label_json:
    label_dict = json.load(label_json)
label_dict = {v: k for k, v in label_dict.items()}
xmls_path = "/data/imagenet/bbox_val/val"

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
# json_root = "test_last/label"

basecam_iou = 0.0
basecam_index = 0
basecam_in = 0

test_number = 2000

score_diff = 0.0

# for img_name in test_img:
# for i in range(len(imgs)):
for i in range(test_number):
    # img_path = os.path.join(test_path, img_name)
    '''read imagenet information'''
    image_name = imgs[i]
    img_id = image_name.split('/')[-1][:-4]
    '''get the label information'''
    xml_pth = os.path.join(xmls_path, img_id + "xml")
    annotation = xml_manage.get_annotation(xml_pth)
    xy_bbox = np.array(annotation['xy_bbox'])
    wh_bbox = np.array(annotation["wh_bbox"])
    class_name = annotation['class_name']
    class_id = int(label_dict[class_name])
    # # image_name = 'scorecut_test/test_multiclass/el4.png'
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

    image = Image.open(image_name)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    im = image.copy()
    input_tensor = transform(image)
    '''draw cub box'''
    # color = (0, 255, 0)
    # draw_box = cv2.rectangle(cv2.imread(image_name), (x1, y1), (x2, y2), color, 5)
    # plt.imshow(draw_box)
    # plt.show()
    # fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    # axs[0].imshow(image)
    # axs[0].axis('off')

    # output = model(dog_cat_image.unsqueeze(0).cuda())
    # print_top_classes(output)

    # cat - the predicted class
    cam_result, featuremap = generate_visualization(input_tensor, im, class_id)
    mask_pred = cv2.resize(featuremap, (im.size[0], im.size[1]), interpolation=cv2.INTER_LINEAR)
    mask_pred = progress_mask(mask_pred)

    '''perturbation test'''

    test_mask = cv2.resize(mask_pred, (224, 224))[None, ...]
    targets = [ClassifierOutputSoftmaxTarget(class_id)]
    pc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # pc = [75]
    for i in range(len(pc)):
        cam_metric = ROADMostRelevantFirst(percentile=pc[i])
        input_tensor = input_tensor[None, ...].to('cuda:0')
        cam_scores, visualizations = cam_metric(input_tensor,
                                                test_mask,
                                                targets,
                                                model,
                                                return_visualization=True,
                                                return_diff=True)
        score_diff += cam_scores[0]

print('cam_diff:{:.4f}'.format(score_diff / (9 * test_number)))

#     if point_game_test:
#         threshold = 0.99
#         # change the threshold of the masked area
#         _, mask_pred_binary_map = cv2.threshold(mask_pred,
#                                                 mask_pred.max() * threshold, 1,
#                                                 cv2.THRESH_TOZERO)
#
#         basecam_in += jug_point_game(mask_pred_binary_map, xy_bbox)
#
#     if bbox_iou_test:
#         threshold = 0.1
#         # change the threshold of the masked area
#         _, mask_pred_binary_map = cv2.threshold(mask_pred,
#                                                 mask_pred.max() * threshold, 1,
#                                                 cv2.THRESH_TOZERO)
#
#         basecam_iou += jug_box_iou(mask_pred_binary_map, wh_bbox)[0]
#         basecam_index += jug_box_iou(mask_pred_binary_map, wh_bbox)[1]
#     '''save result'''
#
#     if save_result:
#         save_path = os.path.join("./outputs", image_name.split("/")[2], 'hila_best')
#         feature_path = os.path.join(save_path, "feature")
#         if os.path.exists(feature_path) is False:
#             os.makedirs(feature_path)
#         cam_path = os.path.join(save_path, "cam")
#         if os.path.exists(cam_path) is False:
#             os.makedirs(cam_path)
#         cv2.imwrite(os.path.join(cam_path, image_name.split("/")[-1]), cam_result[:, :, ::-1])
#         cam_feature = scipy.ndimage.zoom(featuremap, [16, 16], order=0, mode='nearest')
#         plt.imsave(os.path.join(feature_path, image_name.split('/')[-1]), arr=cam_feature)
#
# print("basecam_iou:{:.4f},inbox_iou:{:.4f}".format(basecam_iou / test_number, basecam_iou / basecam_index))
# print('basecam in box:{:.4f}'.format(basecam_in / test_number))

# plt.imshow(cat)
# dog
# generate visualization for class 243: 'bull mastiff'
# dog = generate_visualization(dog_cat_image, im, class_index=1)

# axs[1].imshow(cat)
# axs[1].axis('off')
# axs[1][0].imshow(dog)
# axs[1][0].axis('off')

# plt.show()
