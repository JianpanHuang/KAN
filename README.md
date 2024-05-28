# KAN
This repository contains a demo of the application of efficient Kolmogorov-Arnold Network (KAN) for a regression task. The original implementation of efficient KAN can be found [here](https://github.com/Blealtan/efficient-kan).

The regression task here is curve fitting.

The curve function is: y = a·sin(b·x)+c·cos(d·x), x = 0:0.2:10, as shown below. You may change it to whatever function you would like to fit.

<img width="760" alt="image" src="https://github.com/JianpanHuang/KAN/assets/43700029/b13faadc-f28c-4ec2-8376-1bda193728a7">

The training dataset was created using the matlab code ‘create_dataset.m’ under 'Data' folder.

Network specifics: size(inputlayer, hiddenlayer, outputlayer) = [51, 64, 4].

The input is curve values y with a length of 51, and the output is the coefficients vector [a, b, c, d] with a length of 4.

The loss curves of KAN and MLP after training for 30 epochs are as follows:

<img width="1053" alt="image" src="https://github.com/JianpanHuang/KAN/assets/43700029/a656042a-6e78-4684-b54f-5cb1d3c56483">


The predicted curves by MLP and KAN after training for 30 epochs are as follows:

![Figure_1_res](https://github.com/JianpanHuang/KAN/assets/43700029/cae56653-cea5-4ba2-a273-c3d584fedf6e)



