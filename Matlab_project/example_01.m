clear all;
close all;
clc;

dt = 0.01;
sim_time = 20;
t = 0 : dt : sim_time;

M = 1;    
F = 5* sin(t/10);
X = zeros(2,1,length(t));


for i = 2:length(t)

    
    A = [ 0 1 ; 0 0];
    B = [0; F(i)/M];
    
    X(:,:,i) = A*dt*X(:,:,i-1) + B* dt + X(:,:,i-1);
end

plot(t,reshape(X(2,1,:),[1,2001]), '-b.')




