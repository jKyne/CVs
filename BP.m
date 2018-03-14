clc;
clear;
TrM=loadMNISTImages('train-images-idx3-ubyte');
TeM=loadMNISTImages('t10k-images-idx3-ubyte');
TrL=loadMNISTLabels('train-labels-idx1-ubyte');
TeL=loadMNISTLabels('t10k-labels-idx1-ubyte');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
train_num=60000; %可调
x=TrM(:,1:train_num)';     %x是输入数据集，样本数m*特征数d  
y=zeros(train_num,10);     %y是输出，即标签值,是一个train_num*10维向量，即每个输出是个十维向量，是第几类，第几个数就为1，其它为0
for i=1:train_num
    y(i,TrL(i,1)+1)=1;
end

%获取输入参数的样本数与参数数  
[m,d]=size(x);    %m是x矩阵的行数，表示总共有多少个训练样本。d是矩阵的列数，表示训练数据的特征数。  
hLayer_num=40; %可调%隐层神经元个数
OutL_num=10;      %输出层神经元个数  
  
v=0.01*rand(d,hLayer_num);                %输入层与隐层的权值,v是一个d行hLayer_num列矩阵，输入层d个神经元，隐层hLayer_num个神经元 
w=0.01*rand(hLayer_num,OutL_num);         %隐层与输出层的权值,w是一个hLayer_num行OutL_num列矩阵，隐藏层hLayer_num个神经元，输出层OutL_num个神经元
gamma=rand(hLayer_num,1);            %隐层阈值,gamma是hLayer_num行1列矩阵  
theta=rand(OutL_num,1);       %输出层阈值,theta是OutL_num行1列矩阵  
out=zeros(m,OutL_num);        %输出层输出  
b=zeros(hLayer_num,1);                 %隐层输出  
g=zeros(1,OutL_num);            %均方误差对w求导的参数  
e=zeros(hLayer_num,1);                 %均方误差对v求导的参数  
  
eta=0.1;           %可调             %学习率  
  
  
kn=1;        %迭代的次数  
sn=0;        %同样的均方误差值累积次数  
all_E=0;     %记录每一次迭代的累计误差 

while(1)  
    kn=kn+1;  
    E=0;      %当前迭代的均方误差  
    for i=1:m  
      %计算隐层输出  
      for j=1:hLayer_num  
        alpha=x(i,:)*v(:,j);   %当前一个隐层节点神经元的输入   
        b(j,1)=1/(1+exp(-alpha+gamma(j,1)));  %计算某个隐层节点的输出  
      end  
       %计算输出层输出  
       for j=1:OutL_num    
        beta=b'*w;  
        out(i,j)=1/(1+exp(-beta(1,j)+theta(j,1)));  
       end  
        
        %计算当前一个训练数据的均方误差  
        for j=1:OutL_num 
          A= out(i,:)-y(i,:);
          E=E+sum(A.*A)/2;  
        end  
        %计算w的导数参数g  
          g=out(i,:).*(ones(1,OutL_num)-out(i,:)).*(y(i,:)-out(i,:));  
        %计算v导数参数 e  
          e=w*g'.*b .*(ones(hLayer_num,1)-b);  
          
         %更新v,gamma   
           gamma = gamma - eta *e;         
           v= v + (eta*e*x(i,:))';  
            
          %更新w,theta  
            theta = theta -eta*g';   
            w =w + eta *b*g;  
         end  
%          迭代终止判断  
         if (kn==46) %%%防止E很大前后差<0.001这个条件太苛刻
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
   fprintf('第%d次迭代成功.\n',kn-1);      
end

test_num=600;
x=TeM(:,1:test_num)';     %x是输入测试集，样本数m*特征数d  
y_T=TeL(1:test_num,1);     %y是测试标签值
pred=zeros(test_num,1);   %预测值
out_T=zeros(test_num,OutL_num);        %输出层输出  
b_T=zeros(hLayer_num,1);                 %隐层输出 
    for i=1:test_num  
      %计算隐层输出  
      for j=1:hLayer_num  
        alpha=x(i,:)*v(:,j);   %当前一个隐层节点神经元的输入   
        b_T(j)=1/(1+exp(-alpha+gamma(j,1)));  %计算某个隐层节点的输出  
      end  
       %计算输出层输出  
       for j=1:OutL_num    
        beta=b_T'*w;  
        out_T(i,j)=1/(1+exp(-beta(1,j)+theta(j,1)));  
       end 
      
    end
    
    %判别种类
[a,pred]=max(out_T,[],2);
pred=pred-ones(test_num,1);
accuracy=mean(double(pred == y_T));
fprintf('用BP算法得到模型的预测准确率为：%d .\n',accuracy);                                   
      
figure;
n=1:kn-2;
stem(n,all_E(2:kn-1));
xlabel('迭代次数');
ylabel('均方误差E');
title('均方误差随迭代次数的变化');
    
    
    
    
    
    
