function [tAcc, tErr] = TrainFitness(Function_name,Runno,mlpConfig,solution)

if Function_name=="F1"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("Cancer");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        
        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %         end
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);

        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        tmptAcc(agentNo,1) = mean(mean(Index1 == Index2)) * 100;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
        %         fprintf('Testing tAcc. : %.2f \n',xTrain);
        %         fprintf('Testing tAcc. : %.2f \n',(Index1 == Index2) * 100);
        %         tester=mean(mean(Index1 == Index2)) * 100;
        
        %         figure(2)
        %         hold on
        %          semilogy(tester,'r')
        %         drawnow
        %
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end

if Function_name=="F2"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("Heart");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    parfor agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        
        %Testing with Test Data
        H = logsig(xTrain*wi + repmat(bi,size(xTrain,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTrain,1),1));
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        tmptAcc(agentNo,1) = mean(mean(Index1 == Index2)) * 100;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end

if Function_name=="F3"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);

        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %         end
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;
        
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end

if Function_name=="F4"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID_UNDER");
    
   in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);

        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        
        % fprintf('Testing tAcc. : %.2f \n',xTrain);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
%         fprintf('Index1 tAcc. :  \n',Index1);
%         fprintf('Index2 tAcc. :  \n',Index2);

        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;

%       fprintf('pertAcc tAcc. : %.2f \n',pertAcc);
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc); %test
    tErr = mean(tmptErr);
    
end

if Function_name=="F5"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID_OVER");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        
        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;
        
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end


if Function_name=="F6"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;
        
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end


if Function_name=="F7"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID_UNDER22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;
        
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end


if Function_name=="F8"
    
    [xTrain, tTrain, ~, ~] = DatasetInit("COVID_OVER22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTrain,2),L,O);
        %Feed Forword to Train
        H = logsig(xTrain*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node

        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTrain,[],2);
        % fprintf('Testing tAcc. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumtAcc = sum(Index1 == Index2);
        pertAcc = (sumtAcc/size(xTrain,1))*100;
                
        tmptAcc(agentNo,1) = pertAcc;
        tmptErr(agentNo,1) = mse(tTrain - Y);
        
    end
    
    tAcc = mean(tmptAcc);
    tErr = mean(tmptErr);
    
end

end