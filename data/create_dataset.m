clear all;
ds = 50000; % data size
x=0:0.2:10;
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