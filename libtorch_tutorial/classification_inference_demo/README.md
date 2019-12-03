c++ inference for classification
======
c++环境配置
----
* libtorch1.3.0_cpu
* opencv4.1.1
* cmake 3.5
* g++ 5.4.0

python 环境配置
* [requirements](./requirements.txt)

程序编译和运行:
---------

```
>>>cd ./bin
>>> ./run.sh

```

程序运行的结果:
-----
```
python inference results: tensor([ 1.5317, -3.6304,  0.2571,  0.3478,  1.4500, -0.3430,  0.3210, -1.0825,
        -3.9176, -2.6490], grad_fn=<SliceBackward>)
top-0 label: n02108422 bull mastiff, score: 17.990623
top-1 label: n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier, score: 13.381636
top-2 label: n02109047 Great Dane, score: 12.846677
top-3 label: n02093256 Staffordshire bullterrier, Staffordshire bull terrier, score: 12.175651
top-4 label: n02110958 pug, pug-dog, score: 11.985804
###################################################################
Starting c++ inference.....
c++ load model ok
execution time = 3.23786s
C++ inference results: 1.5317 -3.6304  0.2571  0.3478  1.4500 -0.3430  0.3210 -1.0825 -3.9176 -2.6490
[ Variable[CPUFloatType]{1,10} ]
top-1 label: n02108422 bull mastiff, score: 17.9906
top-2 label: n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier, score: 13.3816
top-3 label: n02109047 Great Dane, score: 12.8467
top-4 label: n02093256 Staffordshire bullterrier, Staffordshire bull terrier, score: 12.1757
top-5 label: n02110958 pug, pug-dog, score: 11.9858
```
