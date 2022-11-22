clc

x_coeffs = [500 700 -450 600 640];
y_coeffs = [300 -200 350 -400 100];
x_ineq_num = 4;
y_ineq_num = 4;
x_ineq_b = 200;
y_ineq_b = 100;
x_lb = 150;
x_ub = 400;
y_lb = 120;
y_ub = 320;
Aeq = [];
beq = [];
ga_time_limit = 10;
avr_ga_iter_num = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_num = size(x_coeffs,2);
y_num = size(y_coeffs,2);
f = -1 * [x_coeffs, y_coeffs];

x_b_arr = x_ineq_b * ones(1,x_ineq_num);
y_b_arr = y_ineq_b * ones(1,y_ineq_num);
b = [x_b_arr, y_b_arr];

xd = -1* eye(x_num);
xv1 = ones(1,x_num - 1);
xd1 = diag(xv1, 1);
A1 = xd + xd1;
yd = -1* eye(y_num);
yv1 = ones(1,y_num - 1);
yd1 = diag(yv1, 1);
A2 = yd + yd1;

A = [A1, zeros(x_num,y_num); zeros(y_num,x_num), A2];
A(x_num,:)=[];
A(x_num + y_num - 1,:)=[];

x_lb_arr = x_lb * ones(1,x_num);
x_ub_arr = x_ub * ones(1,x_num);

y_lb_arr = y_lb * ones(1,y_num);
y_ub_arr = y_ub * ones(1,y_num);

lb = [x_lb_arr, y_lb_arr];
ub = [x_ub_arr, x_ub_arr];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_case = linprog(f, A, b, Aeq, beq, lb, ub);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fun = @(x) fitness(x, x_coeffs, y_coeffs);
nonlcon = @(x) nl_const(x, x_num, y_num);
nvars = x_num + y_num;
%options = optimoptions("ga", "MaxTime", ga_time_limit);
start_point = [100 100 100 100 100 100 100 100 100 100];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nl_ga = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon);
nl_ga_avr_arr = avrage_ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, avr_ga_iter_num);
nl_ps = patternsearch(fun, start_point, A, b, Aeq, beq, lb, ub, nonlcon);
nl_fmc = fmincon(fun, start_point, A, b, Aeq, beq, lb, ub, nonlcon);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_case
nl_ga
nl_ga_avr_arr
nl_ps
nl_fmc
%%%%%%%%%%%%%%%functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nl_ga_avr_arr = avrage_ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon, iter_num)
nl_ga_arr = zeros(iter_num, nvars);
nl_ga_avr_arr = zeros(1,nvars);
for i = 1:iter_num
    nl_ga_arr(i,:) = ga(fun, nvars, A, b, Aeq, beq, lb, ub, nonlcon);
end
for j = 1:nvars
    nl_ga_avr_arr(j) = ( mean( nl_ga_arr(:,j) ) );
end
end

function [c,ceq] = nl_const(x, x_num, y_num)
s = x(1 : x_num);
h = x(x_num + 1: x_num + y_num);
c = s.^2 + h.^2 -40000;
ceq = [];
end

function c = fitness(x, x_coeffs, y_coeffs)
f2 = [x_coeffs, y_coeffs];
c = f2 * x';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
