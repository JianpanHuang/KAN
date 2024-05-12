# RegKAN
This repository contains a demo of a regression task (curve fitting) using an efficient Kolmogorov-Arnold Network (KAN). You can find the implementation of efficient KAN [here](https://github.com/Blealtan/efficient-kan).


Curve function: y = a*sin(b*x)+c*cos(d*x);
x = 0:0.2:10;
Network input = y (size = 51);
Network output = [a,b,c,d] (size = 4);


The dataset was generate by the following matlab code:
clear all;
ds = 50000; % data size
x = 0:0.2:10;
for n = 1:ds
    a = randsample(20,1)/10;
    b = randsample(20,1)/10;
    c = randsample(20,1)/10;
    d = randsample(20,1)/10;
    y = a*sin(b*x)+c*cos(d*x); % y function
    y = awgn(y,30,"measured");
    input(:,n) = y;
    target(:,n) = [a,b,c,d];
end
% figure, plot(x,y)
inputs = input';
targets = target';
csvwrite('inputs.csv', inputs);
csvwrite('targets.csv', targets);


For 20 epochs with a single hiddenlayer(64 neurons), the loss curves of KAN and MLP are as follows:
<img width="1153" alt="image" src="https://github.com/JianpanHuang/RegKAN/assets/43700029/579b4077-4974-40b9-afe2-cd9e1447f877">

