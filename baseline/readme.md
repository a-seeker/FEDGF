### 基线算法

数据集位置：数据集为mnist、fashion-mnist、cifar10，运行代码会自动检测数据集是否存在，不存在则自动下载,下载位置为../../data/*下

数据集格式为： mnist和fashionmnist数据集图片大小28×28，cifar10图片大小为3×32×32

数据集处理脚本：

启动入口处自动根据命令中指定的数据集加载对应的数据集

![image-20240413104543201](C:\Users\yuansd\Desktop\代码审核\exp\baseline\assets\image-20240413104543201.png)

数据集处理函数：

baseline/utils/util.py

![image-20240413104617331](C:\Users\yuansd\Desktop\代码审核\exp\baseline\assets\image-20240413104617331.png)

noniid切分

![image-20240413104646031](C:\Users\yuansd\Desktop\代码审核\exp\baseline\assets\image-20240413104646031.png)

基线算法包含FedAvg、FedProx、DynaFed算法，启动脚本时需设置相关超参，超参默认值已经在fedgf/utils/options.py中设置，根据不同数据集和不同Non-IID划分，超参需要改变，具体设置如下：

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
# FedAvg
python -u server/FedavgServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --flmethod fedavg --seed 2020 --use_wandb --project cifar-ae-1 

# FedProx
python -u server/FedProxServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --flmethod fedprox --seed 2024 --use_wandb --project cifar-ae-1 

# Scaffold
python -u server/ScaffoldServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50 --flmethod scaffold --seed 2022 --use_wandb --project cifar-ae-1 
```

##### 狄利克雷0.2

```
# FedAvg
python -u server/FedavgServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --noniid_type dirichlet --alpha 0.2 --flmethod fedavg --seed 2024 --use_wandb --project cifar-ae-1 

# FedProx
python -u server/FedProxServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --noniid_type dirichlet --alpha 0.2 --flmethod fedprox --seed 2022 --use_wandb --project cifar-ae-1 

# Scaffold
python -u server/ScaffoldServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.2 --flmethod scaffold --seed 2022 --use_wandb --project cifar-ae-1 
```

##### 狄利克雷0.1

```
# FedAvg
python -u server/FedavgServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --noniid_type dirichlet --alpha 0.1 --flmethod fedavg --seed 2020 --use_wandb --project cifar-ae-1 

# FedProx
python -u server/FedProxServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50  --noniid_type dirichlet --alpha 0.1 --flmethod fedprox --seed 2022 --use_wandb --project cifar-ae-1 

# Scaffold
python -u server/ScaffoldServer.py  --dataset cifar10 --num_channels 3  --model cnn --epochs 200 --gpu 0 --num_users 50 --noniid_type dirichlet --alpha 0.1 --flmethod scaffold --seed 2024 --use_wandb --project cifar-ae-1 
```

