## R-Cut: Enhancing Explainability in Vision Transformers with Relationship Weighted Out and Cut
![R-Cut](https://github.com/oniu-uin0/R-Cut/blob/main/rcut.png)

## Usage - Explainability in ViT
There are no extra compiled components in R-Cut and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/oniu-uin0/R-Cut.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## Data preparation

Download and extract ImageNet1k dataset with annotations from
[https://www.image-net.org/](https://www.image-net.org/).
We expect the directory structure to be the following:
```
path/to/imagenet/
  img_val/
  bbox_val/
  ILSVRC2012_devkit_t12/
  train/
  test/
  val/
    n01440764/
      ...JEPG
      
```
## Training
To train baseline ViT on a single node run:

--need to change data path

After training, you can get a pre-trained model weights.
```
python vit_train.py 
```

## Testing
To test R_Cut explainability based on the trained weights on a single node run:

--need to change data path

```
python r_cut.py
```

## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@Article{s24092695,
AUTHOR = {Niu, Yingjie and Ding, Ming and Ge, Maoning and Karlsson, Robin and Zhang, Yuxiao and Carballo, Alexander and Takeda, Kazuya},
TITLE = {R-Cut: Enhancing Explainability in Vision Transformers with Relationship Weighted Out and Cut},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {9},
ARTICLE-NUMBER = {2695},
URL = {https://www.mdpi.com/1424-8220/24/9/2695},
PubMedID = {38732800},
ISSN = {1424-8220},
DOI = {10.3390/s24092695}
```

