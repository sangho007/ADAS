%%

%% example 1 : const
clear;
clc;
t = (0:0.1:30)';

y_true      = 5*ones(size(t));
noise_1     = normrnd(0,0.4,size(y_true));
y_measure   = y_true + noise_1;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.y_measure  = y_measure;
ex_1.y_estimate  = y_estimate;
ex_1.y_true     = y_true;
ex_1.noise      = noise_1;

plot(t,y_true,'r-',t,y_measure,'bo')

% writetable(struct2table(ex_1),'Data/example_Filter_1.txt');
% fclose('all');

%% examle 2 : x^2
clear;
clc;
t = (0:0.1:30)';

y_true      = (t-10).^2./100+1;
noise_1     = normrnd(0,0.4,size(y_true));
y_measure   = y_true + noise_1;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.y_measure  = y_measure;
ex_1.y_estimate  = y_estimate;
ex_1.y_true     = y_true;
ex_1.noise      = noise_1;

plot(t,y_true,'r-',t,y_measure,'bo')

% writetable(struct2table(ex_1),'Data/example_Filter_2.txt');
% fclose('all');

%% example 3 : exp*sine
clear;
clc;
t = (0:0.1:30)';

y_true      = 3*exp(-t/20) .* cos(t*(1));
noise_1     = normrnd(0,0.5,size(y_true));
y_measure   = y_true + noise_1;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.y_measure  = y_measure;
ex_1.y_estimate  = y_estimate;
ex_1.y_true     = y_true;
ex_1.noise      = noise_1;

plot(t,y_true,'r-',t,y_measure,'bo')

% writetable(struct2table(ex_1),'Data/example_Filter_3.txt');
% fclose('all');

%% Kalman filter example 1
clear;
clc
t = (0:0.1:30)';

u = 0.5*sin(t.^2/50);
m = 1;

y_true = zeros(size(t));
for i = 2:length(t)
    y_true(i) = y_true(i-1) + u(i-1)/m + u(i-1)*normrnd(0,0.8);
end

noise_1     = normrnd(0,1.5,size(y_true));
y_measure   = y_true + noise_1;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.u          = u;
ex_1.y_measure  = y_measure;
ex_1.y_estimate  = y_estimate;
ex_1.y_true     = y_true;
ex_1.noise      = noise_1;

plot(t,y_true,'r-',t,y_measure,'bo')

% writetable(struct2table(ex_1),'Data/example06.txt');
% fclose('all');

%% example 7
clear;
clc
dt = 0.1;
t = (0:dt:30)';

u = (ceil(sin(t/2))*2-1);
m = 1;
q1 = 0.01;
q2 = 0.1;
Q = diag([q1 q2]);

x_true = zeros(size(t));
v_true = zeros(size(t));
for i = 2:length(t)
    x_true(i) = x_true(i-1) + v_true(i-1)*dt + normrnd(0,q1);
    v_true(i) = v_true(i-1)*(0.95) + 1*dt/m*u(i-1)*0.97 + normrnd(0,q2);
end

noise_r     = normrnd(0,0.7,size(x_true));
y_measure   = x_true + noise_r;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.u          = u;
ex_1.y_measure  = y_measure;
ex_1.y_estimate = y_estimate;
ex_1.y_true     = x_true;
ex_1.noise      = noise_r;

plot(t,x_true,'r-',t,y_measure,'bo',t,u,'c.')

writetable(struct2table(ex_1),'Data/example07.txt');
fclose('all');
%% example 8
clear;
clc
dt = 0.1;
t = (0:dt:30)';

u = (ceil(sin(t/2))*2-1)*100;
m = 10;
k = 100;
b = 2;
q1 = 0.01;
q2 = 0.1;
Q = diag([q1 q2]);

x_true = zeros(size(t));
v_true = zeros(size(t));
for i = 2:length(t)
    x_true(i) = x_true(i-1) + v_true(i-1)*dt + normrnd(0,q1);
    v_true(i) = v_true(i-1)*(1-b/m) - x_true(i-1)*k/m*dt + 1/m*dt*u(i-1) + normrnd(0,q2);
end

noise_r     = normrnd(0,0.1,size(x_true));
y_measure   = x_true + noise_r;
y_estimate  = zeros(size(t));

ex_1            = {};
ex_1.time       = t;
ex_1.u          = u;
ex_1.y_measure  = y_measure;
ex_1.y_estimate = y_estimate;
ex_1.y_true     = x_true;
ex_1.noise      = noise_r;

plot(t,x_true,'r-',t,y_measure,'bo',t,u/100,'c.')

% writetable(struct2table(ex_1),'Data/example08.txt');
% fclose('all');
