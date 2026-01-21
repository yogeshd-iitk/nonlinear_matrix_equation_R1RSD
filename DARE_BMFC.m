% clc;
clearvars;
close all;

%% input
n=10; 
%% Coefficient matrices
U = orth(randn(n));
V=  orth(randn(n));
A=U*diag((rand(n, 1)+0.1))*V';
G=5*eye(n); 
U = orth(randn(n)); D = diag(rand(n, 1)+0.8); Q = U*D*U'; 
%% Initialization
iter=n*10000;
% armijo parameters
p= 1/8; 
bta= 0.01;
dlta_0=1;
% matrix parameters
I=eye(n);
X=Q;
Y=chol(X)';

N_t_inv = inv(I + G * X);  
N_t_inv_A= N_t_inv*A;
At_X=A'*X;
At_X_N_t_inv_A = At_X*N_t_inv_A;

f_iter=zeros(iter+1,1);
step=zeros(iter,1);
DLTA=zeros(iter,1);
GRAD_IJ=zeros(iter,1);
T_BM=zeros(iter,1);
Iter_rmijo=zeros(iter,1);
f_iter(1) = norm((X-At_X_N_t_inv_A-Q),"fro")^2;
f_x=f_iter(1);
%
%% Algorithm
for m = 1:iter
    isInt = mod(m/100,1) == 0;
    if isInt 
        fprintf('iter = %d, f = %g\n',m,f_iter(m));
        if(f_iter(m)<1e-6)
            ITER=m;
            break
        end
    end

    tic;
%% coordinate selection
    % randomized selection
    %i = randi(n,1);
    %j = randi(n,1);

    % without replacement
    i = mod(ceil(m/n),n);
    if (i==0)
        i = n;
    end
    j = mod(m,n);
    if (j==0)
        j = n;
    end
    
    %% Compute necessary vectors
    e_i = I(:,i);
    e_j = I(:,j);
    %% gradient of the function
    M_1=(X-At_X_N_t_inv_A-Q);
    y_i=Y(:,i);
    g_1=M_1'*y_i+N_t_inv_A* (M_1'*(A'*(-y_i+ X*(N_t_inv*(G*y_i)))));
    g_2=A*(M_1*(A'*(N_t_inv'*y_i)));
    g_3=M_1*y_i+(-g_2+G'*(N_t_inv'*(X*g_2)));
    g_4=g_1+g_3;
    gradf_ij=g_4(j,1);
    GRAD_IJ(m,1)=gradf_ij;
    %
    dlta=dlta_0;
    f=1e8;
    q=0;
    s=sign(gradf_ij);

    while f > (f_x - bta*  dlta *(s)*(gradf_ij)) 
            if dlta < 1e-10
                break
            end
            dlta = p*dlta;
            q=q+1;
            %% Armijo line search
            %
            g_temp=G(:,i);
            y_temp=Y(:,j);
            c_1= 1 - dlta*(s) * (y_temp' * (N_t_inv * g_temp));  
            N_2_inv = N_t_inv +  (N_t_inv *((dlta*(s)/ (c_1)) * g_temp)) * (y_temp' * N_t_inv) ;

            N_2_inv_A = N_t_inv_A + (N_t_inv * ((dlta*(s) / (c_1))* g_temp ))* (y_temp' * N_t_inv_A) ;

            At_X_N_2_inv_A = At_X_N_t_inv_A +  (A'*(X*(N_t_inv * ((dlta*(s)/ (c_1)) * g_temp)))) * (y_temp' * N_t_inv_A) ;

            c_2 = 1 - (dlta*(s) *  N_2_inv(i,:))* (G * Y(j,:)');  

            N_1_inv = N_2_inv + (N_2_inv * (G *((dlta*(s)/ (c_2)) *  Y(:,j))))* N_2_inv(i,:) ;
            N_1_inv_A = N_2_inv_A + (N_2_inv * (G *((dlta*(s)/ (c_2)) *  y_temp))) * N_2_inv_A(i,:) ;

            At_X_N_1_inv_A = At_X_N_2_inv_A + (A'*(X*(N_2_inv * (G * ((dlta*(s)/ (c_2)) * Y(:,j)))))) * N_2_inv_A(i,:) ;

            v_1=N_1_inv * (g_temp);
            v_2=N_1_inv(i,:);
            v_3= N_1_inv_A(i,:);

            c_6 = 1 + dlta^2 * v_1(i,1);

            N_tp1_inv = N_1_inv - ((dlta^2/ (c_6)) * v_1) * v_2 ;
            N_tp1_inv_A = N_1_inv_A - ((dlta^2/ (c_6) )* v_1 )* v_3 ; % N_tp1_inv*A
            %                 
            v_4 = A(i,:)';  
            v_5 = y_temp;
            v_6 = A' * v_5;
            
            X_new=X;
            X_new(i,:)=X_new(i,:)-dlta*(s)*v_5';
            X_new(:,i)=X_new(:,i)-dlta*(s)*v_5;
            X_new(i,i)=X_new(i,i)+dlta^2;
             
            At_X_new= At_X - (dlta*(s)*v_4)* v_5';
            At_X_new(:,i) = At_X_new(:,i)+(- dlta*(s)* v_6+dlta^2 * v_4);

            c_3=v_5'*(N_1_inv*g_temp);
            c_44=N_1_inv*g_temp;
            c_4=c_44(i,1);  
            At_X_new_N_tp1_inv_A = At_X_N_1_inv_A - (dlta*(s)*v_4)* (v_5'*N_1_inv_A) - (dlta*(s)* v_6)* v_3 + (dlta^2 *v_4)* v_3...
                              - (At_X* ((dlta^2/c_6) *v_1)) * v_3  + ((c_3*dlta^3*(s)/ (c_6))*v_4)* v_3...
                              + ((c_4*dlta^3*(s)/ (c_6))* v_6)*v_3  - ((c_4* dlta^4/ (c_6)) * v_4)* v_3 ; 

            resdl=(X_new-At_X_new_N_tp1_inv_A-Q);  
            f = norm(resdl,"fro")^2 ;
            resdl=[];

    end
    DLTA(m,1)=dlta;

    f_x=f;
    f_iter(m+1)=f;
    N_t_inv=N_tp1_inv;
    N_t_inv_A = N_tp1_inv_A;
    Y(i,j)=Y(i,j)-dlta*(s);  
    X=X_new;
    At_X=At_X_new;
    At_X_N_t_inv_A= At_X_new_N_tp1_inv_A;

    f_iter(m+1)=f_x;
     Iter_rmijo(m)=q;
T_BM(m)=toc;
step(m)=dlta/abs(gradf_ij);
end
%% Display results
f_iter=f_iter(1:m);
semilogy(f_iter);
