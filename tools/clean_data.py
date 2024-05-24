import json
import os

home_root = '/home/niu/work/pytorch_learn/deep-learning-for-image-processing/data_set/importance_jari/importance_jari'
remote_root = '/home/y_niu/workspace3/work/data_set/importance_jari/importance_jari'


def save_data_list(root, save_name):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    data_name = root.split("/")[-1]
    data_dict = dict((cls_name, os.listdir(os.path.join(root, cls_name))) for cls_name in os.listdir(root))
    json_data = json.dumps(data_dict, indent=4)
    with open(os.path.join(data_name + save_name + ".json"), 'w') as json_file:
        json_file.write(json_data)


def substrate_data():
    with open("../importance_jariclean.json", "r") as cleaned:
        cleaned_dict = json.load(cleaned)
    with open("importance_jarinotclean.json", "r") as not_cleaned:
        not_cleaned_dict = json.load(not_cleaned)

    for cls_name in cleaned_dict.keys():
        dl = list(set(not_cleaned_dict[cls_name]) - set(cleaned_dict[cls_name]))
        count = 0
        for pic in dl:
            os.remove(os.path.join(remote_root, cls_name, pic))
            count += 1
        print("delete:{}".format(count))


if __name__ == '__main__':
    save_data_list(remote_root, 'notclean')
    substrate_data()