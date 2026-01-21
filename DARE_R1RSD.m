% clc;
clearvars;
close all;
%% input
n=10; 
power_iter=10; 
%% Coefficient matrices
U = orth(randn(n));
V=  orth(randn(n));
A=U*diag((rand(n, 1)+0.1))*V';
G=5*eye(n); 
U = orth(randn(n)); D = diag(rand(n, 1)+0.8); Q = U*D*U'; 
%
%% Initialization
X=Q;
B=chol(X)';
M_1=inv(eye(n)+G*X);
iter=n*1000;
f_iter=zeros(iter+1,1);
T_SD=zeros(iter+1,1);
alpha_SD=zeros(iter,1);
LAM_SD = zeros(iter,1);
options = optimset('TolX',1e-7); 
%%
M_2=A'*X*M_1*A;
M_3= X-M_2-Q;
M_4=A*M_3*A'*M_1';
M_5=G'*M_1'*X;  
f_iter(1)=norm(M_3,"fro")^2;

%% symbolic function expression
syms alpha fc_1 fc_2 fc_3 fc_4 fc_5 fc_6 fc_7 fc_8 b_2

b_1= alpha*b_2+1;
%% fun min 
expr= (alpha*(fc_1) +alpha^2*(fc_2) +(alpha/(b_1))*(fc_3) + (alpha^2/(b_1))*(fc_4) + (alpha^2/(b_1)^2)*(fc_5) ...
    + (alpha^3/(b_1))*(fc_6) + (alpha^3/(b_1)^2)*(fc_7) + (alpha^4/(b_1)^2)*(fc_8));
%
f_1 = matlabFunction(expr, 'Vars', [alpha, fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, fc_8, b_2]);
T_SD(1)=toc;
%% R1RSD algorithm
for l=1:iter
    isInt = mod(l/100,1) == 0;
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
            t_3=B'*((M_3'*u) +M_4'*(- u + M_5'*u )); 
            t_4=M_4*u;
            t_5=B'*((M_3*u) +(- t_4 + M_5*t_4));
            t_6=t_3+t_5;
            y= t_6/norm(t_6);
        end
         %% approx max eigenvalue
            u_eg=B*y;
            t_3eg=B'*((M_3'*u_eg) +M_4'*(- u_eg + M_5'*u_eg )); 
            t_4eg=M_4*u_eg;
            t_5eg=B'*((M_3*u_eg) +(- t_4eg + M_5*t_4eg));
            t_eg=t_3eg+t_5eg;
            lam_eg=(u_eg')*t_eg;
            LAM_SD(l)=lam_eg;

        %% constants
            u=B*y;
            g=M_1'*u;
            v=A'*g;    
            s=A'*u;
            h=M_1*(G*u);
            h_1=G'*(M_1'*u);
            w=  A'*(X*h);         
            b_2_val=  u'*h ;      
            c_1=v'*v;  c_2=s'*s;  c_3=w'*s;  c_4=v'*u;   c_5=u'*s;
            %
            fc_1_val=2*u'*(M_3)*u- 2*s'*(M_3)*v ;
            fc_2_val=(u'*u)^2 - 2*(c_5)*(c_4) + (c_2)*(c_1);
            fc_3_val= 2*w'*(M_3)*v;
            fc_4_val=2*b_2_val*s'*(M_3)*v +  2*(u'*w)*(c_4)  - 2*(c_3)*(c_1);
            fc_5_val=(w'*w)*(c_1);
            fc_6_val=2*b_2_val*(c_5)*(c_4)- 2*b_2_val*(c_2)*(c_1);
            fc_7_val=2*b_2_val*(c_3)*(c_1);
            fc_8_val=b_2_val^2*(c_2)*(c_1);

        %% minimize function
        f=@(alpha) f_1(alpha, fc_1_val, fc_2_val,fc_3_val, fc_4_val,fc_5_val, fc_6_val,fc_7_val, fc_8_val, b_2_val); % fun min

        [alpha_min, f_min] = fminbnd(f,-1,10, options); 
        alpha_SD(l)=alpha_min; 
        f_iter(l+1)=f_iter(l)+f_min;
        if alpha_min~=0

            b_1_val=alpha_min*b_2_val+1;
            %
            M_5=M_5- ((alpha_min/b_1_val)*h_1)*(h'*X)...
                +( alpha_min*h_1 - ((h'*u)*alpha_min^2/b_1_val)*h_1)*u';  
         
            r=(alpha_min-alpha_min^2*b_2_val/b_1_val)*s - (alpha_min/b_1_val)*w;
          
            M_4= M_4+ (alpha_min*(A*u))*((u'*A')*M_1') + (A*r)*((v'*A')*M_1') ...
                - ((M_4+ c_4*alpha_min*A)*((alpha_min/b_1_val)*u) + (A*(c_1*(alpha_min/b_1_val)*r)))*h';
        
            M_2=M_2 + r*v';
           
            X=X+ ((alpha_min*u)*u');
           
            M_1=M_1- (alpha_min/b_1_val*h)*(u'*M_1);
            
            M_3= X-M_2-Q;
            %
            if alpha_min>0
                B=cholupdate(B',sqrt(alpha_min)*u,'+')';
            else
                B=cholupdate(B',sqrt(abs(alpha_min))*u,'-')';
            end
        end
        T_SD(l)=toc;
end

%% plot
    f_iter=f_iter(1:l+1);
    T_SD=T_SD(1:ITER);
    semilogy(f_iter); 