import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# 其对应的cotegory有7种
# INSECT_NAMES = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

# 构造类别对应的索引字典
def get_insect_name(INSECT_NAMES):
    """获取类别索引字典
    return a dict,
        {'Boerner': 0,
         'Leconte': 1,
         'Linnaeus': 2,
         'acuminatus': 3,
         'armandi': 4,
         'coleoptera': 5,
         'linnaeus': 6
        }
    """
    insect_category2id = {}  # 从类别到id的一个映射
    for i, item in enumerate(INSECT_NAMES):  # 利用enumerate自带的索引序号进行映射
        insect_category2id[item] = i

    return insect_category2id


# 传入类别索引字典和xml数据地址进行xml解析
# 返回xml中解析到的所有目标数据
def get_annotation(data_path):
    '''
        category2id: 类别索引字典
        data_path: 数据地址（单个xml的数据的相对地址——1.xml）

        return records
                (这是一个list，包含其中返回的字典数据)
    '''
    fid = data_path[:-4]  # id：既是xml的，通常也是img的
    tree = ET.parse(data_path)

    objs = tree.findall('object')  # 返回所有的object对象（标签）
    img_w = tree.find('size').find('width').text  # 通过text属性调用xml指定标签的数据
    img_h = tree.find('size').find('height').text
    # print('Image Width:', img_w, 'px , Image Height:', img_h, 'px')

    # 预置我们需要从xml中读取的数据格式——这里的bbox采用xywh的数据格式存储
    # 根据objs的个数，分配此时解析xml需要多少个bbox
    xy_bbox = np.zeros((len(objs), 4, 2), dtype=np.float32)
    wh_bbox = np.zeros((len(objs), 4), dtype=np.float32)  # 4：表示bbox的数据: xywh
    gt_class = np.zeros((len(objs), 1), dtype=np.int32)  # 1：对应类别索引
    is_difficult = np.zeros((len(objs), 1), dtype=np.int32)  # 1：对应是否识别困难

    # 预置后数据格式之后，开始遍历objs，依次读取其中的数据，存入到预置的数据格式中
    for i, obj in enumerate(objs):  # 利用enumerate返回自带索引，来方便多个obj的索引和遍历
        # 类别读取
        category_name = obj.find('name').text  # 获取类别名
        # category_id = category2id[category_name]  # 获取对应的类别索引
        # gt_class[i] = category_id  # 添加识别类别索引

        # 识别是否困难
        # _difficult = int(obj.find('difficult').text)  # 获取识别难度
        # is_difficult[i] = _difficult  # 读取目标识别是否困难

        # bbox读取
        xmin = int(obj.find('bndbox').find('xmin').text)  # 将读取的左上点的x数据转换为float数据
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)  # 右下点
        ymax = int(obj.find('bndbox').find('ymax').text)

        xy_bbox[i] = np.array([[xmin, ymin],
                               [xmax, ymin],
                               [xmax, ymax],
                               [xmin, ymax]])
        '''when do the point test should add one channel'''
        # [xmin, ymax]])[:, None,:]
        # 将读取的bbox数据转换为：xywh的格式存入gt_bbox中
        bbox_center_x = (xmax + xmin) / 2.0
        bbox_x = xmin
        bbox_center_y = (ymax + ymin) / 2.0
        bbox_y = ymin
        bbox_w = (xmax - xmin) + 1.0
        bbox_h = (ymax - ymin) + 1.0
        # 写入gt_bbox,wh_bbox[i]
        # wh_bbox = [bbox_center_x, bbox_center_y, bbox_w, bbox_h]  # 这里等价于将gt_bbox[i,:] = [....]
        wh_bbox[i] = [bbox_x, bbox_y, bbox_w, bbox_h]
    # 将当前xml中的所有obj遍历完后，对数据进行一个汇总
    # 具体实践时的数据解析和汇总可以做适当的调整
    xml_rcd = {
        'image_name': fid + '.JPEG',  # 图片名称
        'image_id': fid,  # 图片id
        # 'difficult': is_difficult,  # 数据识别困难情况
        # 'categoty_id': gt_class,  # 识别对象的类别情况
        'class_name': category_name,
        'xy_bbox': xy_bbox,
        'wh_bbox': wh_bbox  # 识别对象的bbox情况
    }

    # 返回解析完归总后的xml内容
    return xml_rcd


if __name__ == "__main__":
    xmls_path = "/data/imagenet/bbox_val/val"
    imgs_pth = '/data/imagenet/img_val/'

    imgs = glob.glob(imgs_pth + '*.JPEG')
    imgs = sorted(imgs, key=lambda x: x.split('_')[-1][:-4])
    for idex in range(20):
        # img = imgs[idex]
        img = os.path.join(imgs_pth, 'ILSVRC2012_val_00001738.JPEG')
        pic = cv2.imread(img)

        img_name = img.split("/")[-1][:-4]
        xml_pth = os.path.join(xmls_path, img_name + "xml")
        annotation = get_annotation(xml_pth)
        xy_bboxs = np.array(annotation['xy_bbox'])
        # wh_bbox = np.array(annotation['wh_bbox'])
        gt_mask_pic = np.zeros((pic.shape[0], pic.shape[1]))
        for xy_bbox in xy_bboxs:
            area = np.array(xy_bbox).reshape((-1, 1, 2)).astype(np.int32)
            gt_mask_pic = cv2.fillPoly(gt_mask_pic, [area], 1)
        '''show bbox'''
        point_top, point_bot = (int(xy_bbox[0][0]), int(xy_bbox[0][1])), \
                               (int(xy_bbox[2][0]), int(xy_bbox[3][1]))
        cv2.rectangle(pic, point_top, point_bot, (0, 255, 0), 1, 4)

        plt.imshow(pic)
        plt.show()
        # cv2.namedWindow("rectangle")
        # cv2.imshow('rectangle', pic)
        # cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        # cv2.destroyAllWindows()
