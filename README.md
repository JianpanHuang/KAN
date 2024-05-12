# RegKAN
This repository contains a demo of the application of efficient Kolmogorov-Arnold Network (KAN) for a regression task (RegKAN). The original implementation of efficient KAN can be found [here](https://github.com/Blealtan/efficient-kan).

The regression task here is curve fitting.

The curve function is: y = a·sin(b·x)+c·cos(d·x), x = 0:0.2:10.

The training dataset was created using the matlab code ‘create_dataset.m’ under 'Data' folder.

Network specifics: size(inputlayer, hiddenlayer, outputlayer) = [51, 64, 4].

The loss curves of KAN and MLP after training for 20 epochs are as follows:

<img width="1153" alt="image" src="https://github.com/JianpanHuang/RegKAN/assets/43700029/579b4077-4974-40b9-afe2-cd9e1447f877">

