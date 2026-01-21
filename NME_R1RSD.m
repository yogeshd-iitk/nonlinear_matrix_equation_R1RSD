%clc;
clearvars;
close all;
%% input
n=10; 
power_iter=10; 
iter=n*100; 
% coefficient matrices
U = orth(randn(n)); V=  orth(randn(n));Da = diag(0.1*rand(n, 1)); A = U*Da*V';
U = orth(randn(n)); Dy = diag(rand(n, 1)+0.4); Y = U*Dy*U';
Q=Y'*Y; 

%% Initialize
X=Q; 
X_inv=inv(X);
B=chol(X)';
M_1=(X+(A'/X)*A-Q);
M_3= A*M_1*A';
f_iter=zeros(iter+1,1);
alpha_SD=zeros(iter,1); 
LAM_SD = zeros(iter,1); 
T_SD=zeros(iter,1);
f_iter(1)=norm(M_1,"fro")^2;
options = optimset('TolX',1e-7);
%% function expression
syms alpha fc_1 fc_2 fc_3 fc_4 fc_5

    mu = (-alpha)/(alpha+1); 
    %% fun min 
    expr= (alpha*(fc_1) +mu*(fc_2) +alpha^2*(fc_3) + mu*alpha*(fc_4) + mu^2*(fc_5));
    f_1 = matlabFunction(expr, 'Vars', [alpha, fc_1, fc_2, fc_3, fc_4, fc_5]);

for l=1:iter
    isInt = mod(l/50,1) == 0;
    if isInt 
        fprintf('iter = %d, f = %g\n',l,f_iter(l));
        if(f_iter(l)<1e-6)
            ITER=l;
            break
        end
    end
    tic;
    %% power method
    y=randn(n,1);
    y = y/norm(y);
    for m=1:power_iter
        u=B*y;
        t_2=2*B'*(M_1*u - X_inv*(M_3*(X_inv*u)));
        y=t_2/norm(t_2);
    end
    %% approx max eigenvalue
    u_eg=B*y;
    t_eg=2*B'*(M_1*u_eg - X_inv*(M_3*(X_inv*u_eg)));
    lam_eg=(u_eg')*t_eg;
    LAM_SD(l)=lam_eg;
    %% 
     u=B*y;
     z=B'\y;
     v=A*u;
     w= A'*z;
     s=A*w;
     fc_1_val = 2*u'*M_1*u;
     fc_2_val = 2*z'*M_3*z;
     fc_3_val = (u'*u)^2;
     fc_4_val = 2*(u'*w)^2;
     fc_5_val = (w'*w)^2;

   %% update
    f=@(alpha) f_1(alpha, fc_1_val, fc_2_val, fc_3_val, fc_4_val, fc_5_val); 

    [alpha_min, f_min] = fminbnd(f,-1,10,options);
    f_iter(l+1)=f_iter(l)+f_min;
    alpha_SD(l)=alpha_min; 
    mu_min = (-alpha_min)/(alpha_min+1);
    M_3 =M_3 + (alpha_min*v)*v' +  (mu_min*s)*s';
    M_1 =  M_1 +  (alpha_min*u)*u' + (mu_min*w)*w';
    X=X+ (alpha_min*u)*u';
    X_inv= X_inv+(mu_min*z)*z';
    if alpha_min >0
        B=cholupdate(B',sqrt(alpha_min)*u,'+')';
    else
        B=cholupdate(B',sqrt(abs(alpha_min))*u,'-')';
    end
    T_SD(l)=toc;
end

%% Plot results
    f_iter=f_iter(1:l);
    semilogy(f_iter); 