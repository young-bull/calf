# Read me

使用多种深度学习框架实现常见、经典的ML、DL算法，对比参照。

## 目录命名

`<Paradigm>/<问题>/<数据>/<算法>-<框架>-<其他>.ipynb`

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
    * Caffe2 & Pytorch
    * MXNet & gluon
        * 无GPU： `pip install mxnet`
        * 有GPU且安装了CUDA： `pip install mxnet-cu<xx>` // xx 为 CUDA 的版本
* GO 
    * `jupyter notebook`
