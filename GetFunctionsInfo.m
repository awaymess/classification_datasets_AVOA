% This function containts full information and implementations of the
% datasets

% lb is the lower bound: lb=[lb_1,lb_2,...,lb_d]
% up is the uppper bound: ub=[ub_1,ub_2,...,ub_d]
% dim is the number of variables (dimension of the problem)
%%

function [lb,ub,dim,fobj,inp,hidn,outp] = GetFunctionsInfo(F,NoOfHidden)

switch F      
    case 'F1'
        fobj = @MLP_Cancer;
        lb=-10;
        ub=10;
        %dim=209+1;
        inp=9;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
    case 'F2'
        fobj = @MLP_Heart;
        lb=-10;
        ub=10;
        %dim=1081;
        inp=13;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);

     case 'F3'
        fobj = @MLP_COVID;
        lb=-100;
        ub=100;
        inp=8;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp); %364 dim
        
     case 'F4'
        fobj = @MLP_COVID_UNDER;
        lb=-100;
        ub=100;
        inp=8;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
     case 'F5'
        fobj = @MLP_COVID_OVER;
        lb=-100;
        ub=100;
        inp=8;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
     case 'F6'
        fobj = @MLP_COVID22;
        lb=-100;
        ub=100;
        inp=7;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
      case 'F7'
        fobj = @MLP_COVID22nodep;
        lb=-100;
        ub=100;
        inp=7;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
      case 'F8'
        fobj = @MLP_COVID_UNDER22;
        lb=-100;
        ub=100;
        inp=7;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
      case 'F9'
        fobj = @MLP_COVID_OVER22;
        lb=-100;
        ub=100;
        inp=7;
        hidn=NoOfHidden;
        outp=2;
        % Searching Space Size: (input x hidden) + (hidden * output) + (hidden + output)
        % (Weight for Input -> Hidden) + (Weight for Hidden -> Output) + (Bias for Hidden and Output)
        dim = (inp*hidn)+(hidn*outp)+(hidn+outp);
        
end

end

function o=MLP_Cancer(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("Cancer");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;

end


function o=MLP_Heart(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("Heart");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;

end

function o=MLP_COVID(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node

end

fitness = mse(e);
o=fitness;

end

function o=MLP_COVID_UNDER(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID_UNDER");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;
end

function o=MLP_COVID_OVER(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID_OVER");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node
    
end

fitness = mse(e);
o=fitness;
end

function o=MLP_COVID22(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID22");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node

end

fitness = mse(e);
o=fitness;

end

function o=MLP_COVID22nodep(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID22nodep");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node

end

fitness = mse(e);
o=fitness;

end

function o=MLP_COVID_UNDER22(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID_UNDER22");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node

end

fitness = mse(e);
o=fitness;

end

function o=MLP_COVID_OVER22(solution,mlpConfig)

[xTrain, tTrain, ~, ~] = DatasetInit("COVID_OVER22");

in = mlpConfig.inp; % Number of Input Node
L = mlpConfig.hidn; %Number of Hidden Node
O = mlpConfig.outp; % Number of Output Node

% Assign Weight

fitness = 0;

[wi, bi, wo, bo] = MLPWeightInit(solution,size(xTrain,2),L,O); % Weight Assign

% Evaluation
for i = 1:size(xTrain,1)
    
    %Feed Forword to Train
    H = logsig(xTrain(i,:)*wi + bi); %Output from Hidden Node
    Y = logsig(H*wo + bo); %Output from Output Node
    
    e(i,:) = tTrain(i,:) - Y; %Error from Output Node

end

fitness = mse(e);
o=fitness;

end

