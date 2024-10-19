### FedAEF

### 1、数据集

数据集位置：数据集为mnist、fashion-mnist、cifar10，运行代码会自动检测数据集是否存在，不存在则自动下载,下载位置为../../data/*下

数据集格式为： mnist和fashionmnist数据集图片大小28×28，cifar10图片大小为3×32×32

数据集处理脚本：

启动入口处自动根据命令中指定的数据集加载对应的数据集

![image-20240413103549894](C:\Users\yuansd\Desktop\代码审核\exp\fedaef\assets\image-20240413103549894.png)

数据集处理函数：

![image-20240413103513064](C:\Users\yuansd\Desktop\代码审核\exp\fedaef\assets\image-20240413103513064.png)

当加载完成后，执行Non-IID数据划分方法：

![image-20240413103834735](C:\Users\yuansd\Desktop\代码审核\exp\fedaef\assets\image-20240413103834735.png)

FedAEF算法基于自动编码器实现，启动文件为trainae_v3.py，启动脚本时需设置相关超参，超参默认值已经在fedaef/utils/options.py中设置，根据不同数据集和不同Non-IID划分，超参需要改变，具体设置如下：

### 2、运行方法

#### **mnist数据集**

##### 病态Non-IID划分

```
python -u trainae_v3.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --seed 2023 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-mnist-niid
```

##### 狄利克雷0.2

```
python -u trainae_v3.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 500 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-mnist-0.2
```

##### 狄利克雷0.1

```
python -u trainae_v3.py --dataset mnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 500 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-mnist-0.1
```



#### Fashion-Mnist数据集

##### 病态Non-IID划分

```
python -u trainae_v3.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --seed 2023 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-fashionmnist-niid
```

##### 狄利克雷0.2

```
python -u trainae_v3.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 2023 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-fashionmnist-0.2
```

##### 狄利克雷0.1

```
python -u trainae_v3.py --dataset fashionmnist --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 2023 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-fashionmnist-0.1
```



#### CIfar-10数据集

##### 病态Non-IID划分

```
python -u trainae_v3.py --dataset cifar --model cnn --num_channels 3 --epochs 200 --gpu 0 --num_users 50 --seed 2024 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-cifar-niid
```

##### 狄利克雷0.2

```
python -u trainae_v3.py --dataset cifar --model cnn --num_channels 3 --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --seed 3 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-cifar-0.1
```

##### 狄利克雷0.1

```
python -u trainae_v3.py --dataset cifar --model cnn --num_channels 3 --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --seed 2023 --local_ep2 20 --lr2 0.002 --use_wandb --project fedtc-cifar-0.2
```

### 3、代码注释

每节对应的注释已经标注

### 4、结果

代码的运行结果都会出现在算法根目录下，结果文件格式为:

算法名_non-iid划分方式_数据集_日期.xls

大论文中表5.11~表5.16和图5.7~图5.9对应代码位置

![image-20240410001835141](C:\Users\yuansd\Desktop\代码审核\exp\fedaef\assets\image-20240410001835141.png)
