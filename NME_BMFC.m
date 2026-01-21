% clc;
clearvars;
close all;
%% input
n=10;
iter=n*100;
% coefficient matrices
U = orth(randn(n)); V=  orth(randn(n));Da = diag(0.1*rand(n, 1)); A = U*Da*V';
U = orth(randn(n)); Dy = diag(rand(n, 1)+0.4); Z = U*Dy*U';
Q=Z'*Z; 

%% Initialization
dlta_0=0;
I=eye(n);
X=Q;
Y=chol(X)'; 
Y_inv =inv(Y);
X_inv = inv(X);
M_1=(X+(A'/X)*A-Q);

f_iter=zeros(iter+1,1);
T_BM=zeros(iter,1);
delta_BM=zeros(iter,1);
f_iter(1)=norm(M_1,"fro")^2;
options = optimoptions('fminunc','OptimalityTolerance',1e-6, 'Display', 'off');
%% function expression
syms dlta c_1 c_2 fc_1 fc_2 fc_3 fc_4 fc_5 fc_6 fc_7 fc_8 fc_9 fc_10 fc_11 fc_12 fc_13
    alpha = dlta/(1-dlta*c_1);
      bta=  alpha^2*c_2;
    %% fun min 
    expr= (dlta*(fc_1) +alpha*(fc_2) +bta*(fc_3) + dlta^2*(fc_4) + dlta^3*(fc_5)...
        + alpha*dlta*(fc_6) +bta*dlta*(fc_7) +dlta^4*(fc_8) + alpha*dlta^2*(fc_9) + bta*dlta^2*(fc_10)...
        +alpha^2*(fc_11) + alpha*bta*(fc_12) + bta^2*(fc_13));
    f_1 = matlabFunction(expr, 'Vars', [dlta, c_1, c_2, fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, fc_8, fc_9, fc_10, fc_11, fc_12, fc_13]);
for l=1:iter
    isInt = mod(l/100,1) == 0;
    if isInt 
        fprintf('iter = %d, f = %g\n',l,f_iter(l));
        if(f_iter(l)<1e-6)
            break
        end
    end
    tic;
    %% coordinate selection
        % randomized selection
            %i = randi(n,1);
            %j = randi(n,1);
    
        % without replacement
            i = mod(ceil(l/n),n);
            if (i==0)
                i = n;
            end
            j = mod(l,n);
            if (j==0)
                j = n;
            end
    
    %% 
    %% Compute necessary vectors
    e_i = I(:,i);
    e_j = I(:,j);
    %% constants evaluation
    % vectors
    n_5=Y(:,j);
    v_i= Y_inv(:,i);
    v_j= Y_inv(j,:)'; 
    w_j=A'*v_j;
    s_i=A'*X_inv(:,i);
    % scalars
    c_1_val=Y_inv(j,i);
    c_2_val= v_i'*v_i;
    c_3= n_5(i,1);
    c_4=w_j(i,1);
    c_5=s_i(i,1);
    c_6=n_5'*n_5;
    c_7=n_5'*w_j;
    c_8=n_5'*s_i;
    c_9=s_i'*w_j;
    c_10=w_j'*w_j;
    c_11= s_i'*s_i;
    % equation constants
    fc_1_val  = -4* n_5'*M_1(:,i);
    fc_2_val  = 4* s_i'*(M_1*w_j);
    fc_3_val  = 2*w_j'*(M_1*w_j);
    fc_4_val  = 2*(M_1(i,i)+c_3^2+c_6);
    fc_5_val  = -4*c_3;
    fc_6_val  = -4*(c_5*c_7+c_4*c_8);
    fc_7_val  = -4*c_4*c_7;
    fc_8_val  = 1;
    fc_9_val  = 4*c_4*c_5;
    fc_10_val = 2*c_4^2;
    fc_11_val = 2*(c_9^2+c_10*c_11);
    fc_12_val = 4*c_9*c_10;
    fc_13_val = c_10^2;
     %% update
     f=@(dlta) f_1(dlta, c_1_val, c_2_val, fc_1_val, fc_2_val, fc_3_val, fc_4_val, fc_5_val, fc_6_val, fc_7_val, fc_8_val, ...
         fc_9_val, fc_10_val, fc_11_val, fc_12_val, fc_13_val); % fun min
         % clc;
        [dlta_min, f_min] = fminunc(f,dlta_0, options);
        if (f_min<-0.01)
            break
        end

    %%  gradient  coefficient at (i,j)
        d=4*(M_1*Y(:,i)-X_inv*(A*(M_1*((A')*(Y_inv(i,:)')))));
        grad_coff=d(j,1);

    %% Update
        f_iter(l+1)=f_iter(l)+f_min;
        delta_BM(l)=dlta_min/grad_coff;
        alpha_min = dlta_min/(1-dlta_min*c_1_val);
        bta_min=  alpha_min^2*c_2_val;
        %
        M_1= M_1+ (alpha_min*w_j)*s_i'+ (alpha_min*s_i)*w_j'+ (bta_min*w_j)*w_j';
        M_1(i,:)=M_1(i,:)-dlta_min*n_5';
        M_1(:,i)=M_1(:,i)-dlta_min*n_5;
        M_1(i,i)=M_1(i,i)+dlta_min^2;
        
        Y(i,j) =Y(i,j)- dlta_min ; 

        c_yinv = dlta_min/(1+Y_inv(j,i));
        X_inv= X_inv +  (c_yinv*v_j)*(v_i'*Y_inv) + (Y_inv'*v_i)*(c_yinv*v_j') + ((c_yinv)^2*(v_i'*v_i)*v_j)*v_i';

        Y_inv=Y_inv+ (c_yinv*v_i)*(v_j');
        %
        X(i,:)=X(i,:)-dlta_min*n_5';
        X(:,i)=X(:,i)-dlta_min*n_5;
        X(i,i)=X(i,i)+dlta_min^2;
      
    T_BM(l)=toc;
end
%% Plot result
ITER=l;
T_BM=T_BM(1:l);
f_iter=f_iter(1:l+1);
semilogy(f_iter);