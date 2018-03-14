clc;
clear;
TrM=loadMNISTImages('train-images-idx3-ubyte');
TeM=loadMNISTImages('t10k-images-idx3-ubyte');
TrL=loadMNISTLabels('train-labels-idx1-ubyte');
TeL=loadMNISTLabels('t10k-labels-idx1-ubyte');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
train_num=60000; %�ɵ�
x=TrM(:,1:train_num)';     %x���������ݼ���������m*������d  
y=zeros(train_num,10);     %y�����������ǩֵ,��һ��train_num*10ά��������ÿ������Ǹ�ʮά�������ǵڼ��࣬�ڼ�������Ϊ1������Ϊ0
for i=1:train_num
    y(i,TrL(i,1)+1)=1;
end

%��ȡ����������������������  
[m,d]=size(x);    %m��x�������������ʾ�ܹ��ж��ٸ�ѵ��������d�Ǿ������������ʾѵ�����ݵ���������  
hLayer_num=40; %�ɵ�%������Ԫ����
OutL_num=10;      %�������Ԫ����  
  
v=0.01*rand(d,hLayer_num);                %������������Ȩֵ,v��һ��d��hLayer_num�о��������d����Ԫ������hLayer_num����Ԫ 
w=0.01*rand(hLayer_num,OutL_num);         %������������Ȩֵ,w��һ��hLayer_num��OutL_num�о������ز�hLayer_num����Ԫ�������OutL_num����Ԫ
gamma=rand(hLayer_num,1);            %������ֵ,gamma��hLayer_num��1�о���  
theta=rand(OutL_num,1);       %�������ֵ,theta��OutL_num��1�о���  
out=zeros(m,OutL_num);        %��������  
b=zeros(hLayer_num,1);                 %�������  
g=zeros(1,OutL_num);            %��������w�󵼵Ĳ���  
e=zeros(hLayer_num,1);                 %��������v�󵼵Ĳ���  
  
eta=0.1;           %�ɵ�             %ѧϰ��  
  
  
kn=1;        %�����Ĵ���  
sn=0;        %ͬ���ľ������ֵ�ۻ�����  
all_E=0;     %��¼ÿһ�ε������ۼ���� 

while(1)  
    kn=kn+1;  
    E=0;      %��ǰ�����ľ������  
    for i=1:m  
      %�����������  
      for j=1:hLayer_num  
        alpha=x(i,:)*v(:,j);   %��ǰһ������ڵ���Ԫ������   
        b(j,1)=1/(1+exp(-alpha+gamma(j,1)));  %����ĳ������ڵ�����  
      end  
       %������������  
       for j=1:OutL_num    
        beta=b'*w;  
        out(i,j)=1/(1+exp(-beta(1,j)+theta(j,1)));  
       end  
        
        %���㵱ǰһ��ѵ�����ݵľ������  
        for j=1:OutL_num 
          A= out(i,:)-y(i,:);
          E=E+sum(A.*A)/2;  
        end  
        %����w�ĵ�������g  
          g=out(i,:).*(ones(1,OutL_num)-out(i,:)).*(y(i,:)-out(i,:));  
        %����v�������� e  
          e=w*g'.*b .*(ones(hLayer_num,1)-b);  
          
         %����v,gamma   
           gamma = gamma - eta *e;         
           v= v + (eta*e*x(i,:))';  
            
          %����w,theta  
            theta = theta -eta*g';   
            w =w + eta *b*g;  
         end  
%          ������ֹ�ж�  
         if (kn==46) %%%��ֹE�ܴ�ǰ���<0.001�������̫����
             break
         end
         if(abs(all_E(kn-1)-E)<0.1)  
            sn=sn+1;
            all_E(kn)=E; 
            if(sn==3)  
               break;  
             end  
          else  
             all_E(kn)=E;  
             sn=0;  
         end
   fprintf('��%d�ε����ɹ�.\n',kn-1);      
end

test_num=600;
x=TeM(:,1:test_num)';     %x��������Լ���������m*������d  
y_T=TeL(1:test_num,1);     %y�ǲ��Ա�ǩֵ
pred=zeros(test_num,1);   %Ԥ��ֵ
out_T=zeros(test_num,OutL_num);        %��������  
b_T=zeros(hLayer_num,1);                 %������� 
    for i=1:test_num  
      %�����������  
      for j=1:hLayer_num  
        alpha=x(i,:)*v(:,j);   %��ǰһ������ڵ���Ԫ������   
        b_T(j)=1/(1+exp(-alpha+gamma(j,1)));  %����ĳ������ڵ�����  
      end  
       %������������  
       for j=1:OutL_num    
        beta=b_T'*w;  
        out_T(i,j)=1/(1+exp(-beta(1,j)+theta(j,1)));  
       end 
      
    end
    
    %�б�����
[a,pred]=max(out_T,[],2);
pred=pred-ones(test_num,1);
accuracy=mean(double(pred == y_T));
fprintf('��BP�㷨�õ�ģ�͵�Ԥ��׼ȷ��Ϊ��%d .\n',accuracy);                                   
      
figure;
n=1:kn-2;
stem(n,all_E(2:kn-1));
xlabel('��������');
ylabel('�������E');
title('�����������������ı仯');
    
    
    
    
    
    
