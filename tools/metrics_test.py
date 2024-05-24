import cv2
import numpy as np
import matplotlib.pyplot as plt


def test_single_iou(box1, box2):
    axmin, axmax, aymin, aymax = box1[0], (box1[0] + box1[2]), box1[1], (box1[1] + box1[3])
    bxmin, bxmax, bymin, bymax = box2[0], (box2[0] + box2[2]), box2[1], (box2[1] + box2[3])
    width = min(axmin, bxmin) + (axmax - axmin) + (bxmax - bxmin) - max(axmax, bxmax)
    height = min(aymin, bymin) + (aymax - aymin) + (bymax - bymin) - max(aymax, bymax)
    if width <= 0 or height <= 0:
        return 0
    else:
        return width * height / ((bymax - bymin) * (bxmax - bxmin) + (aymax - aymin) * (axmax - axmin) - width * height)


def test_mask_iou(gt, pre):
    total = gt + pre
    i = np.sum(total >= 2)
    u = np.sum(total >= 2) + np.sum(total == 1)
    return i / u


def jug_point_game(mask_pred_map, xy_bbox):
    contours, _ = cv2.findContours((mask_pred_map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # for contour in contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        box2 = [x, y, w, h]
        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
        """Determine whether the point is in the area"""
    center_point = (int(x + w / 2), int(y + h / 2))
    # center_point = np.unravel_index(np.argmax(mask_pred_map), mask_pred_map.shape)
    results = 0.0
    for bb in xy_bbox:
        results += cv2.pointPolygonTest(bb, center_point, False)
    if results >= 0.0:
        return 1
    else:
        return 0


def jug_single_point_game(mask_pred_map, xy_bbox):
    contours, _ = cv2.findContours((mask_pred_map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # for contour in contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        #         # Determine whether the point is in the area
        #         center_point = (int(x + w / 2), int(y + h / 2))
        #         cls, point_pol = point_in_out(image_name, json_root, center_point)
        #         if point_pol == 1.0 or point_pol == 0.0:
        #             in_list.append(cls)
        # in_dict = Counter(in_list)
        # print(in_dict)
        box2 = [x, y, w, h]
        estimated_bbox = [x, y, x + w, y + h]
        color1 = (0, 0, 255)
        """Determine whether the point is in the area"""
    center_point = (int(x + w / 2), int(y + h / 2))
    # center_point = np.unravel_index(np.argmax(mask_pred_map), mask_pred_map.shape)
    result = cv2.pointPolygonTest(xy_bbox, center_point, False)
    if result == 1.0 or result == 0.0:
        return 1
    else:
        return 0


def jug_single_box_iou(map, bbox):
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
        if test_single_iou(bbox, box2) > 0:
            return test_single_iou(bbox, box2), 1, estimated_bbox
        else:
            return 0, 0, estimated_bbox


def jug_box_iou(map, xy_bbox):
    contours, _ = cv2.findContours((map * 255).astype(np.uint8),
                                   cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # here maybe can find many minicontours
    pre_mask_pic = np.zeros(map.shape)
    for c in contours:
        # c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
        area = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
        pre_mask_pic = cv2.fillPoly(pre_mask_pic, [area], 1)
        # plt.imshow(pre_mask_pic)
        # plt.show()

    gt_mask_pic = np.zeros(map.shape[:2])
    for bbox in xy_bbox:
        area = np.array(xy_bbox).reshape((-1, 1, 2)).astype(np.int32)
        gt_mask_pic = cv2.fillPoly(gt_mask_pic, [area], 1)

    """determine the IOU of the two boxes"""
    if test_mask_iou(gt_mask_pic, pre_mask_pic) > 0:
        return test_mask_iou(gt_mask_pic, pre_mask_pic), 1, estimated_bbox
    else:
        return 0, 0, estimated_bbox
