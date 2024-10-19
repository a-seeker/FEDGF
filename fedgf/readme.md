### FedGF

### 1、数据集

数据集为mnist、fashion-mnist、cifar10，运行代码会自动检测数据集是否存在，不存在则自动下载,下载位置为../../data/*下

数据集位置：数据集为mnist、fashion-mnist、cifar10，运行代码会自动检测数据集是否存在，不存在则自动下载,下载位置为../../data/*下

数据集格式为： mnist和fashionmnist数据集图片大小28×28，cifar10图片大小为3×32×32

数据集处理脚本：

启动入口处自动根据命令中指定的数据集加载对应的数据集

![image-20240413103936976](C:\Users\yuansd\Desktop\代码审核\exp\fedgf\assets\image-20240413103936976.png)

数据集处理函数：

![image-20240413104207515](C:\Users\yuansd\Desktop\代码审核\exp\fedgf\assets\image-20240413104207515.png)

当加载完成后，执行Non-IID数据划分方法：

![image-20240413104047738](C:\Users\yuansd\Desktop\代码审核\exp\fedgf\assets\image-20240413104047738.png)

FedAEF算法基于生成对抗网络实现，启动文件为train_merge.py，启动脚本时需设置相关超参，超参默认值已经在fedgf/utils/options.py中设置，根据不同数据集和不同Non-IID划分，超参需要改变，具体设置如下：

### 2、运行方法

#### **mnist数据集**

##### 病态Non-IID划分

```
python -u train_merge.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --seed 2023 --use_wandb --project fedgf-mnist
```

##### 狄利克雷0.2

```
python -u train_merge.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 2023 --use_wandb --project fedgf-mnist
```

##### 狄利克雷0.1

```
python -u train_merge.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 2023 --use_wandb --project fedgf-mnist
```



#### Fashion-Mnist数据集

##### 病态Non-IID划分

```
python -u train_merge.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --seed 2023 --use_wandb --project fedgf-fashionmnist
```

##### 狄利克雷0.2

```
python -u train_merge.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 2023 --use_wandb --project fedgf-fashionmnist
```

##### 狄利克雷0.1

```
python -u train_merge.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 2023 --use_wandb --project fedgf-fashionmnist
```



#### CIfar-10数据集

##### 病态Non-IID划分

```
python -u train_merge.py --dataset cifar --num_channels 3 --model cnn --epochs 200 --gpu 0 --num_users 50 --seed 2023 --use_wandb --project fedgf-cifar
```

##### 狄利克雷0.2

```
python -u train_merge.py --dataset cifar --num_channels 3 --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 2023 --use_wandb --project fedgf-cifar
```

##### 狄利克雷0.1

```
python -u train_merge.py --dataset cifar --num_channels 3 --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 2023 --use_wandb --project fedgf-cifar
```

### 3、代码注释

每节对应的注释已经标注

### 4、结果

代码的运行结果都会出现在算法根目录下，结果文件格式为:

算法名_non-iid划分方式_数据集_日期.xls

大论文中表5.2~表5.7和图5.4~图5.6对应代码位置

![image-20240410001357721](C:\Users\yuansd\Desktop\代码审核\exp\fedgf\assets\image-20240410001357721.png)

