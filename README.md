# RegKAN
This repository contains a demo of a regression task (curve fitting) using an efficient Kolmogorov-Arnold Network (KAN). You can find the implementation of efficient KAN [here](https://github.com/Blealtan/efficient-kan).

The training dataset was created using the matlab code ‘create_dataset.m’.

Curve function:

x = 0:0.2:10;

y = a·sin(b·x)+c·cos(d·x);

Network input = y (size = 51);

Network output = [a,b,c,d] (size = 4);



For 20 epochs with a single hiddenlayer(64 neurons), the loss curves of KAN and MLP are as follows:

<img width="1153" alt="image" src="https://github.com/JianpanHuang/RegKAN/assets/43700029/579b4077-4974-40b9-afe2-cd9e1447f877">

