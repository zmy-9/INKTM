# Input-Aware Neural Knowledge Tracing Machine
ICPR Workshop'2021: Input-Aware Neural Knowledge Tracing Machine.
(Tensorflow implementation for INKTM)

The codes are put into the file INKTM_github!

This is the code for the paper: [Input-Aware Neural Knowledge Tracing Machine](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_25)  

If you find this code is useful for your research, please cite as:
```
Moyu Zhang, Xinning Zhu, and Yang Ji. 2021. Input-Aware Neural Knowledge Tracing Machine. In the Proceedings of Pattern Recognition, ICPR International Workshops and Challenges (ICPR 2021), 345-360.
```

## Setups
* Python 3.6+
* Tensorflow 1.14.0
* Scikit-learn 0.21.3
* Numpy 1.17.2

## How to run model
### If you want to preprocess data, you can do as below:
```
python3 encode.py
```
Take ASSISTments2009 dataset as an example, you just need to download the original dataset and set the path of dataset.
### If you want to predict students' answer, you can do as below:
```
python3 main.py
```

(If you have any questions, please contact me on time. My E-mail is zhangmoyu@bupt.cn.)
