import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from swin_vit_model import swin_tiny_patch4_window7_224 as create_model
# from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         # transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # read class_indict
    json_path = './importance_jari.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    cls_pic = dict((name, []) for name in class_indict.values())
    # create model
    model = create_model(num_classes=10).to(device)

    # load model weights
    weights_path = "./weights_no/jari_swin/model-100.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # prediction
    model.eval()
    # load image
    image_path = "/home/niu/下载/dataset/our/Frames"
    assert os.path.exists(image_path), "file: '{}' dose not exist.".format(image_path)
    test_img = os.listdir(image_path)
    count = 0
    for img_name in test_img:
        img_path = os.path.join(image_path, img_name)
        img = Image.open(img_path)
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        cls_name = class_indict[str(predict_cla)]
        cls_pic[cls_name].append(img_name)
        count += 1
        if count % 5000 == 0:
            print("already processed{}".format(count))
    pic_json = json.dumps(cls_pic, indent=4)
    with open("move.json", 'w') as json_file:
        json_file.write(pic_json)
        # print_res = "class:{}  prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                            predict[predict_cla].numpy())
        # plt.title(print_res)
        # print(print_res)
        # plt.show()


if __name__ == '__main__':
    main()
