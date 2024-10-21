# FLDS
FLDS: Differentially Private Federated Learning with Double Shufflers

本代码主要使用Python 3.8， TensorFlow框架编写，以下提供了服务器环境配置说明、参数设置以及代码执行指南。

环境配置

操作系统要求
Windows 10
处理器要求
11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
安装必要的包
使用pip（Python的包管理器）来安装所有必要的包。首先，确保已经安装了Python和pip，并使用anaconda创建了虚拟文件。

安装必要的包
可以使用pip来安装所有必要的包。在conda虚拟环境中安装以下的包，建议使用指定版本，新版本有可能会对部分函数进行删改：

absl-py                  1.1.0    
astor                    0.8.1    
cached-property          1.5.2    
certifi                  2020.6.20
cycler                   0.11.0   
dataclasses              0.8      
gast                     0.2.2    
google-pasta             0.2.0    
grpcio                   1.46.3   
h5py                     3.1.0    
importlib-metadata       4.8.3    
Keras-Applications       1.0.8    
Keras-Preprocessing      1.1.2    
kiwisolver               1.3.1    
Markdown                 3.3.7    
matplotlib               3.3.4    
mpmath                   1.2.1    
numpy                    1.19.4   
opt-einsum               3.3.0    
Pillow                   8.4.0    
pip                      21.2.2   
protobuf                 3.19.4   
pyparsing                3.0.9    
python-dateutil          2.8.2    
setuptools               59.6.0   
six                      1.16.0   
sympy                    1.1.1    
tensorboard              1.15.0   
tensorflow-cpu           1.15.0   
tensorflow-cpu-estimator 1.15.1   
termcolor                1.1.0    
tqdm                     4.15.0   
typing_extensions        4.1.1    
Werkzeug                 2.0.3    
wheel                    0.37.1   
wincertstore             0.2      
wrapt                    1.14.1   
zipp                     3.6.0

安装代码实例：pip install numpy==1.19.4

TensorFlow的CPU版本可以使用pip直接安装，版本如上所示。如果想使用GPU版本，则需要注意GPU、Python版本和TensorFlow版本对应。


参数设置


默认参数配置如下：
客户端数量：默认设置为6000
epoch轮数：默认设置为10
iters：每轮epoch包括的迭代次数，默认设置为6
每次iter的客户端数量：默认每次iter有1000个客户参与训练
学习速率：初始设置为0.1，在5个epoch后线性降低到0.05，然后固定为0.05

在main.py文件中，主要调整以下超参数来进行实验：
隐私预算 (epsilon)：总体隐私预算默认配置为0.3，搜索范围为{0.05, 0.1, 0.15, 0.2, 0.3}，单个维度的隐私预算需根据文章公式计算，文档中提供了部分整体隐私预算和单个维度的隐私预算的关系，见“FLDS隐私预算参照表.xlsx”。

在fedbase.py文件中，主要调整以下超参数来进行实验：
降维比：默认值为1/50 搜索范围为{1/10, 1/50, 1/157}
调整语句包括：
sum_sample_clients =20
generater_random_matrix(1000, 7850, 20, 157, self.epsilon,self.clip_C)
具体调整数据需要根据降维比修改


代码执行

FLDS中不同数据集的代码是单独的文件，代码文件名进行了标注，对比方案默认MNIST代码，如运行fashion-MNIST，需要将FLDS中的数据集复制并替换掉MNIST数据集
在安装了所有必要的包并设置了正确的超参数后，您可以使用以下命令来执行代码，超参数可以使用下面方式修改或者在main.py中修改，而降维比只能在fedbase.py中修改：

无剪切代码执行：python main.py  --optimizer="flds1"   --epsilon=0.1.545   --learning_rate=0.1  --norm=0.2 - --mechanism="laplace"
无剪切对比方案：代码也在文件“FLDS-MNIST-无剪切”中，将上面的--optimizer="flds1"替换为--optimizer="dpsgd"等，可选择的优化器见子文件flearn中的trainer文件。
有剪切代码运行：python main.py  --MAD_n =3.0  --optimizer="clip_flds1"   --epsilon=0.1.545   --learning_rate=0.1  --norm=0.2 - --mechanism="laplace"
对比方案代码运行：python main.py  --optimizer="dpsgd"   --epsilon=0.3   --learning_rate=0.1  --mechanism="gaussian"
