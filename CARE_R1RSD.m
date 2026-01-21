% clc;
clearvars;
close all;
%% input data 
n=10;
% coefficient matrices
U = orth(randn(n));
V=  orth(randn(n));
A=U*(diag(rand(n, 1)+0.1))*V';
U = orth(randn(n)); Dg = diag(rand(n, 1)+0.8); G = U*Dg*U';
U = orth(randn(n)); Dh = diag(randi(2,n, 1)); H = U*Dh*U';

%% Initialization 
X = eye(n);
B= chol(X)'; % eye(n); %
M_1=(X*G*X-A'*X-X*A-H); % (G-A'-A-H); %
M_2=(G*X-A)*M_1; % (G-A)*M_1; % X=I; % 
iter=n*100;
f_iter=zeros(iter+1,1);
T_SD=zeros(iter,1);
options = optimset('TolX',1e-8); 
f_iter(1)=norm(M_1,"fro")^2;
%% function expression
syms alpha fc_1 fc_2 fc_3 fc_4
%% fun min 
        expr= (alpha*(fc_1) +alpha^2*(fc_2)+alpha^3*(fc_3)+ alpha^4*(fc_4));
        f_1 = matlabFunction(expr, 'Vars', [alpha, fc_1, fc_2, fc_3, fc_4]);
%% R1RSD algorithm 
for l=1:iter
    isInt = mod(l/100,1) == 0;
    if isInt %
        fprintf('iter = %d, f = %g\n',l,f_iter(l));
        if(f_iter(l)<1e-6)
            break
        end
    end
    tic;
        %% power method
        y=ones(n,1);
        for m=1:10
            u=B*y;
            t_2=B'*((M_2+M_2')*u);
            y=t_2/norm(t_2);
        end
        %% constants evaluation
        u=B*y;
        v=G*u;
        c_1=u'*v;
        w=X*v;
        s=A'*u;
        x=w-s;
        r=G*(X*u);
        q=G*(X*x);
        p=A*u;
        c_2=u'*u;
        c_3=u'*x;
        %
        fc_1_val=4*x'*(M_1*u);
        fc_2_val=2*c_1*(u'*(M_1*u))+2*(c_2)*(x'*x)+2*(c_3)*(c_3);
        fc_3_val=4*c_1 *(c_2)* (c_3);
        fc_4_val=c_1^2* (c_2)*(c_2);

        %% update
        f=@(alpha) f_1(alpha, fc_1_val, fc_2_val,fc_3_val, fc_4_val); 

        [alpha_min, f_min] = fminbnd(f,-1,10,options) ; % calcuting alpha through line search

        f_iter(l+1)=f_iter(l)+f_min;

        M_2 = M_2 + (alpha_min*v)*(u'*M_1) +  (alpha_min*r + (c_2* alpha_min^2)*v - alpha_min*p)*x'...
            +  (alpha_min*q +(c_3* alpha_min^2)*v+ (alpha_min^2*c_1)*r + (c_2* alpha_min^3*c_1)*v -  A*(alpha_min*x) - A*(alpha_min^2*c_1*u))*u';

        M_1=M_1 +(alpha_min*u)*x'+ (alpha_min*x + (alpha_min^2*c_1)*u)*u';

        X=X+ (alpha_min*u)*u';
        if alpha_min >0
            B=cholupdate(B',sqrt(alpha_min)*u,'+')';
        else
            B=cholupdate(B',sqrt(abs(alpha_min))*u,'-')';
        end
    T_SD(l)=toc;
end
ITER=l;
f_iter=f_iter(1:l+1);
semilogy(f_iter); % Plot with log scale on y-axis


