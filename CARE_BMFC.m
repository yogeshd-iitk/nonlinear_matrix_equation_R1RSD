% clc;
clearvars;
close all;
%% input
n=10;

% coefficient matrices
U = orth(randn(n));
V=  orth(randn(n));
A=U*(diag(rand(n, 1)+0.1))*V';
U = orth(randn(n)); Dg = diag(rand(n, 1)+0.8); G = U*Dg*U';
U = orth(randn(n)); Dh = diag(randi(2,n, 1)); H = U*Dh*U';

%% Initialization 
I =eye(n);
Y=I;
X=I;
M_5=(X*G*X-A'*X-X*A-H); 
iter=n*1000;
f_iter=zeros(iter+1,1);
T_BM=zeros(iter,1);
f_iter(1)=norm(M_5,"fro")^2;
dlta_0=0;
I_ii=zeros(n,n);
options = optimoptions('fminunc','OptimalityTolerance',1e-6, 'Display','off');
%% function expression
syms dlta fc_1 fc_2 fc_3 fc_4 fc_5 fc_6 fc_7 fc_8
%% fun min 
    expr= (dlta*(fc_1) +dlta^2*(fc_2)+dlta^3*(fc_3)+ dlta^4*(fc_4)+ dlta^5*(fc_5) +dlta^6*(fc_6)+dlta^7*(fc_7)+ dlta^8*(fc_8));
    f_1 = matlabFunction(expr, 'Vars', [dlta fc_1 fc_2 fc_3 fc_4 fc_5 fc_6 fc_7 fc_8]);

%% BMFC Algorithm
for l = 1:iter
    isInt = mod(l/100,1) == 0;
    if isInt 
        fprintf('iter = %d, f = %g\n',l,f_iter(l));
        if(f_iter(l)<1e-6)
            break
        end
    end
    ITER=l;
    tic;
    %% coordinate selection
    % randomized slection
        % i = randi(n,1);
        % j = randi(n,1);

    % without replacement
        i = mod(ceil(l/n),n);
        if (i==0)
            i = n;
        end
        j = mod(l,n);
        if (j==0)
            j = n;
        end
   
    %% Compute necessary vectors
    e_i = I(:,i);
    e_j = I(:,j);

 %% constants evaluation
     n_1= A(i,:)';
     n_5=Y(:,j);
     n_3=A'*n_5;
     temp_1= G(:,i); 
     temp_2= G*n_5;
     n_6=X*(temp_1);
     n_7= X*(temp_2);
     c_1=n_5'*(temp_2);
     c_2= n_5'*(temp_1);
     c_3=G(i,i);
     %
     %
     Tm_12=n_5*n_6';
     Tm_13=n_1*n_5';
     Tm_14=-Tm_12+Tm_13;
     tm_15=-n_7+n_3;
     %
     M_1=Tm_14+Tm_14';
     M_1(:,i)=M_1(:,i)+tm_15;
     M_1(i,:)=M_1(i,:)+tm_15';
     %
     %
     tm_23=c_2*n_5+n_6-n_1;
     %
     M_2=(c_3*n_5)*n_5';
     M_2(:,i)= M_2(:,i)+c_1*e_i+tm_23;
     M_2(i,:)=M_2(i,:)+tm_23';
     %
     %
     M_3=I_ii;
     M_3(i,i)=-2*c_2;
     M_3(:,i)= M_3(:,i)-c_3*n_5;
     M_3(i,:)= M_3(i,:)-c_3*n_5';
     %
     % 
     M_4=I_ii;
     M_4(i,i)=c_3;
     %
     %
     tm_11=M_1(:,i)';
     tm_12=M_1(i,:);
     tm_12(i)=[];
     tm_1= [tm_11 tm_12];
     %
     tm_21=M_2(:,i)';
     tm_22=M_2(i,:);
     tm_22(i)=[];
     tm_2= [tm_21 tm_22];
     %
     tm_31=M_3(:,i)';
     tm_32=M_3(i,:);
     tm_32(i)=[];
     tm_3= [tm_31 tm_32];
     %
     tm_51=M_5(:,i)';
     tm_52=M_5(i,:);
     tm_52(i)=[];
     tm_5= [tm_51 tm_52];
     %
     %
     fc_1_val = 2*trace(M_5*M_1);
     fc_2_val = 2*trace(M_2*M_5) + trace(M_1*M_1);
     fc_3_val = 2*sum((M_3.*M_5),"all")+ 2*sum((M_2.*M_1),"all"); 
     fc_4_val = 2*c_3*M_5(i,i) + 2*sum((tm_3.*tm_1),"all") + sum((M_2.*M_2),"all");
     fc_5_val = 2*c_3*M_1(i,i) + 2*sum((tm_3.*tm_2),"all");
     fc_6_val = 2*c_3*M_2(i,i) + sum((tm_3.*tm_3),"all");
     fc_7_val = 2*c_3*M_3(i,i); 
     fc_8_val = c_3^2 ; 

 %% update
     f=@(dlta) f_1(dlta, fc_1_val, fc_2_val, fc_3_val, fc_4_val, fc_5_val, fc_6_val, fc_7_val, fc_8_val); % fun min

        [dlta_min, f_min] = fminunc(f,dlta_0, options);

        f_iter(l+1)=f_iter(l)+f_min;
        
        M_5=M_5 + dlta_min*(M_1) +  dlta_min^2 * (M_2) + dlta_min^3*(M_3) ;
        M_5(i,i)=M_5(i,i) +  dlta_min^4 *(c_3);

        Y(i,j) = Y(i,j) - dlta_min; 

        X(i,i)=X(i,i)+dlta_min^2;
        X(:,i)= X(:,i)-dlta_min*n_5;
        X(i,:)= X(i,:)-dlta_min*n_5';
    T_BM(l)=toc;
end
%% save and plot data
T_BM=T_BM(1:l,:);
f_iter=f_iter(1:l+1);
semilogy(f_iter); % Plot with log scale on y-axis
