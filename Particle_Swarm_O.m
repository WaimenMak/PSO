function Particle_Swarm_O(popsize,w,c_1,c_2) %粒子群算法
if nargin<1
    popsize = 10; % 种群大小
end
if nargin <2 
    w = 0.7298;
end
if nargin <3 
    c_1 = 1.5;
end
if nargin <4 
    c_2 = 1.5;
end




% bounds = [-5*ones(5,1) 10*ones(5,1)];
% bounds = [-3.0 12.1;4.1 5.8];
% bounds = [-3.0 12.1;0 10];
% bounds = [-1,1;-1,1];
% bounds = [-15*ones(10,1) 30*ones(10,1)];%ackley
% bounds = [0*ones(10,1) pi*ones(10,1)]; %mich
% bounds = [-5*ones(10,1) 10*ones(10,1)];%rosen
bounds = [-4.1*ones(10,1) 6.4*ones(10,1)];%rast
% bounds = [-480*ones(10,1) 750*ones(10,1)];%griewank
% bounds = [-8*ones(4,1) 12.5*ones(4,1)];
% bounds = [-100*ones(2,1) 100*ones(2,1)];%easom
% bounds = [0 1;0 1];
% bounds = [0 1200;0 1200;-0.55 0.55;-0.55 0.55];
% bounds = [-5 5;-5 5];
% bounds = [-100*ones(10,1) 100*ones(10,1)];%trid
% bounds = [-500*ones(20,1) 500*ones(20,1)];%schw
% bounds = [-5*ones(20,1) 10*ones(20,1)];


[dim,~] = size(bounds);
cost = 1e4;   %计算成本
decpop = zeros(popsize,1);   
X_pop = zeros(popsize,dim);        
variable = zeros(1,dim+1);     %保存的变量

V = zeros(popsize,dim);        %s速度矩阵

T = ceil(cost/popsize);

best_mat = zeros(popsize,dim+1);  %记录历史优值和对应位置

best_y = zeros(1,T);


for generation = 1:T
    if generation == 1
        %initialize pop
        for i = 1:popsize
%             if i <= popsize/2     %一半对角一半不对角
%                 for j = 1:dim
%                     X_pop(i,j) =  ((bounds(j,2) - bounds(j,1))*rand + bounds(j,1));
%                 end
%             else 
%                X_pop(i,:) =  ((bounds(:,2) - bounds(:,1))*rand + bounds(:,1))';   %对角线初始化
%             end
            
            for j = 1:dim
                    X_pop(i,j) =  ((bounds(j,2) - bounds(j,1))*rand + bounds(j,1));
            end
            
%             X_pop(i,:) =  ((bounds(:,2) - bounds(:,1))*rand + bounds(:,1))';   %对角线初始化，只能在对角线上搜索了
            best_mat(i,1) = object_function(X_pop(i,:));  %最优值
            best_mat(i,2:dim+1) = X_pop(i,:);                    %最优位置
        end
    else
        [X_pop,V] = update_pop(X_pop,V,best_mat,popsize,w,c_1,c_2,bounds,dim);
%         X_pop = update_pop2(X_pop,V,best_mat,popsize,w,c_1,c_2,bounds,dim);   %速度矩阵不保留历史纪录,这种方法没什么用
        for k =1:popsize          %更新bestmat
            max_adapt = object_function(X_pop(k,:));
            if max_adapt > best_mat(k,1)
                best_mat(k,1) = max_adapt;
                best_mat(k,2:dim+1) = X_pop(k,:);
            end
        end
  
    end
    
    %计算更新后种群中最好的值
    for i = 1:popsize
        [decpop(i)] = object_function(X_pop(i,:));
    end
    
    optimal = max(decpop);
    [rol,~] = find(decpop == optimal);
    best_y(1,generation) = optimal;
    if generation == 1
        variable = [optimal, X_pop(rol(1),:)];
    elseif optimal > variable(1)
        variable = [optimal, X_pop(rol(1),:)];
    end
%     if mod(generation,10)== 1 && dim <=2 
%     visualization(X_pop(:,1),X_pop(:,2),decpop,bounds);
%     title(['迭代次数为n=' num2str(generation)]);
%     plot3(X_pop(rol(1),1),X_pop(rol(1),2),optimal,'o','LineWidth',2);
%     pause;
%     hold off;
%     end
end

    variable
    fprintf('The optimal is --->>%5.4f\n',-variable(1));
%     fprintf('The last optimal is --->>%5.4f\n',optimal);
    
    i = 1:T;
    plot(i,best_y);
    xlabel('Generation');
    ylabel('Optimal');
    title('Convergency');
    
end

function [f_value] = object_function(entity)

X_ = entity;
% f_value = -(X_(1)^2 + X_(2)^2);
% f_value = 21.5+X_(1)*sin(4*pi*X_(1))+X_(2)*sin(20*pi*X_(2));
% f_value = -1*griewank(X_);
% f_value = -1*easom(X_);
% f_value = -1*mich(X_);
% f_value = -1*ackley(X_);
% f_value = -1*rosen(X_);
f_value = -1*rast(X_);
% f_value = -1*hump(X_);
% f_value = -1*trid(X_);
% f_value = colville(X_);
% f_value = -1*Fun1(X_,1);
% f_value = Fun2(X_,5);
% f_value = -1*schw(X_);
% f_value = -1*zakh(X_);

end

function all_best = calculate_best(best_mat,dimension)  %计算所有中的最优
    opt = max(best_mat(:,1));
    [r,~] = find(best_mat(:,1) == opt);
    all_best = best_mat(r(1),2:dimension+1);
    
end


function gbest = calculate_gbest(best_mat,num,popsize,dimension)
    k = 2;
    if num ~= 1 && num ~= popsize
        comp_mat = [best_mat(num-k/2,:);best_mat(num,:);best_mat(num+k/2,:)];
    elseif num == 1
        comp_mat = [best_mat(popsize+(num-k/2),:);best_mat(num,:);best_mat(num+k/2,:)];
    else
        comp_mat = [best_mat(num-k/2,:);best_mat(num,:);best_mat(num+k/2 - popsize,:)] ;   
    end
    best = max(comp_mat(:,1));
    [rol,~] = find(comp_mat(:,1) == best);
    gbest = comp_mat(rol(1),2:dimension+1);
    
end





function [newpop,new_V] = update_pop(X_pop,V,best_mat,popsize,w,c_1,c_2,bounds_,dim)
    boss = calculate_best(best_mat,dim);
    for i  = 1:popsize
        p_best = best_mat(i,2:dim+1);
        g_best = calculate_gbest(best_mat,i,popsize,dim);
        V(i,:) = w*V(i,:) + c_1*rand()*(p_best - X_pop(i,:))+ c_2*rand()*(g_best - X_pop(i,:)) + 1.5*rand*(boss - X_pop(i,:));%环图加一个全部个体最好那个
%         V(i,:) = w*V(i,:) + c_1*rand()*(p_best - X_pop(i,:))+ c_2*rand()*(g_best - X_pop(i,:));%环图
%         V(i,:) = w*V(i,:) + c_1*rand()*(p_best - X_pop(i,:)) + 1.5*rand*(boss - X_pop(i,:));    %完全图
        X_pop(i,:) = X_pop(i,:) + V(i,:);
        for j = 1:dim
            if X_pop(i,j) > bounds_(j,2)
                X_pop(i,j) = bounds_(j,2);
            elseif X_pop(i,j) < bounds_(j,1)
                X_pop(i,j) = bounds_(j,1);
            end
        end
    end
    newpop = X_pop;
    new_V = V;
end

function [newpop] = update_pop2(X_pop,V,best_mat,popsize,w,c_1,c_2,bounds_,dim)
    boss = calculate_best(best_mat,dim);
    for i  = 1:popsize
        p_best = best_mat(i,2:dim+1);
        g_best = calculate_gbest(best_mat,i,popsize,dim);
        V(i,:) = w*V(i,:) + c_1*rand()*(p_best - X_pop(i,:))+ c_2*rand()*(g_best - X_pop(i,:)) + 1.5*rand*(boss - X_pop(i,:));
%         V(i,:) = w*V(i,:) + c_1*rand()*(p_best - X_pop(i,:)) + c_2*rand*(boss - X_pop(i,:));
        X_pop(i,:) = X_pop(i,:) + V(i,:);
        for j = 1:dim
            if X_pop(i,j) > bounds_(j,2)
                X_pop(i,j) = bounds_(j,2);
            elseif X_pop(i,j) < bounds_(j,1)
                X_pop(i,j) = bounds_(j,1);
            end
        end
    end
    newpop = X_pop;
end



function visualization(dim_1,dim_2,decpop,bounds)

[x,y] = meshgrid(bounds(1,1):0.01:bounds(1,2),bounds(2,1):0.01:bounds(2,2));
% z = -(x.^2 + y.^2);
% [row,col] = size(x);
% z = zeros(row,col);

% for i = 1:row
%     for j = 1:col
%         z(i,j) = branin([x(i,j),y(i,j)]);
%     end
% end
z = 21.5+x.*sin(4*pi.*x)+y.*sin(20*pi.*y);

surfl(x,y,z);
% view(0,0);
hold on;
plot3(dim_1,dim_2,decpop,'r*');

end

