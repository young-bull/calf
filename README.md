# Read me

使用多种深度学习框架实现常见、经典的ML、DL算法，对比参照。

## 目录命名

`<Dataset-Paradigm>/<Framework>-<Model/Net.Algorithm>.ipynb`

## 环境准备

* 推荐使用虚拟环境
    * `virtualenv -p python3 myvenv`
    * `source myvenv/bin/activate`
* 必要的Package
    * `pip install numpy scipy pandas matplotlib ipython jupyter jupyter_contrib_nbextensions`
        * `-i https://mirrors.ustc.edu.cn/pypi/web/simple` 可选不同的镜像加速
    * `jupyter contrib nbextension install --sys-prefix`
* ML.DL Framework
    * Tensorflow
        * `pip install tensorflow` or `pip install tensorflow-gpu`
    * Caffe2 & Pytorch
        * https://pytorch.org/
    * MXNet & gluon
        * `pip install mxnet` or `pip install mxnet-cu<xx>` // xx 为 CUDA 的版本: 80,90,91
* GO 
    * `jupyter notebook`
