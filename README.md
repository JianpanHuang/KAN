# KAN
This repository demonstrates the application of efficient Kolmogorov-Arnold Network (KAN) in a regression (curve fitting) task. The original KAN can be found [here](https://github.com/KindXiaoming/pykan), while the original efficient KAN can be found [here](https://github.com/Blealtan/efficient-kan).

The curve function here is: y = a·sin(b·x)+c·cos(d·x), x = 0:0.2:10, as shown below.

**You may change it to whatever function you would like to fit.**

The training dataset was created using the matlab code ‘create_dataset.m’ under 'Data' folder.

Network specifics: size(inputlayer, hiddenlayer, outputlayer) = [51, 100, 4].

The input is curve values y with a length of 51, and the output is the coefficients vector [a, b, c, d] with a length of 4, as shown below.

<img width="1111" alt="image" src="https://github.com/JianpanHuang/KAN/assets/43700029/b406021e-5d43-490d-98fc-7ce5347c1421">

The loss curves of KAN and MLP after training for 30 epochs are as follows:

<img width="1111" alt="image" src="https://github.com/JianpanHuang/KAN/assets/43700029/683a1557-b01a-4204-88ff-13e7d8290301">

The predicted curves by MLP and KAN after training for 30 epochs are as follows:

<img width="555" alt="image" src="https://github.com/JianpanHuang/KAN/assets/43700029/2c39d50e-b48a-42e0-91e4-d48db2590109">



