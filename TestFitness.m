function [Acc, Err] = TestFitness(Function_name,Runno,mlpConfig,solution)

if Function_name=="F1"
    
    [~, ~, xTest, tTest] = DatasetInit("Cancer");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        
        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %         end
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);

        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        tmpAcc(agentNo,1) = mean(mean(Index1 == Index2)) * 100;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
        %         fprintf('Testing ACC. : %.2f \n',xTest);
        %         fprintf('Testing ACC. : %.2f \n',(Index1 == Index2) * 100);
        %         tester=mean(mean(Index1 == Index2)) * 100;
        
        %         figure(2)
        %         hold on
        %          semilogy(tester,'r')
        %         drawnow
        %
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end

if Function_name=="F2"
    
    [~, ~, xTest, tTest] = DatasetInit("Heart");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    parfor agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        
        %Testing with Test Data
        H = logsig(xTest*wi + repmat(bi,size(xTest,1),1));
        Y = logsig(H*wo + repmat(bo,size(xTest,1),1));
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        tmpAcc(agentNo,1) = mean(mean(Index1 == Index2)) * 100;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end

if Function_name=="F3"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);

        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %         end
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
        
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end

if Function_name=="F4"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID_UNDER");
    
   in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);

        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        
        % fprintf('Testing ACC. : %.2f \n',xTest);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
%         fprintf('Index1 ACC. :  \n',Index1);
%         fprintf('Index2 ACC. :  \n',Index2);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
        
%         fprintf('perAcc ACC. : %.2f \n',perAcc);
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end

if Function_name=="F5"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID_OVER");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        
        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
        
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end


if Function_name=="F6"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
        
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end


if Function_name=="F7"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID_UNDER22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node
        
        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
        
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end


if Function_name=="F8"
    
    [~, ~, xTest, tTest] = DatasetInit("COVID_OVER22");
    
    in = mlpConfig.inp; % Number of Input Node
    L = mlpConfig.hidn; %Number of Hidden Node
    O = mlpConfig.outp; % Number of Output Node
    
    for agentNo = 1:Runno
        
        % Assign Weight
        [wi, bi, wo, bo] = MLPWeightInit(solution(agentNo,:),size(xTest,2),L,O);
        %Feed Forword to Train
        H = logsig(xTest*wi + bi); %Output from Hidden Node
        Y = logsig(H*wo + bo); %Output from Output Node

        %Performance of Testing
        [tmp,Index1] = max(Y,[],2);
        [tmp,Index2] = max(tTest,[],2);
        % fprintf('Testing ACC. : %.2f \n',mean(mean(Index1 == Index2)) * 100);
        
        sumAcc = sum(Index1 == Index2);
        perAcc = (sumAcc/size(xTest,1))*100;
                
        tmpAcc(agentNo,1) = perAcc;
        tmpErr(agentNo,1) = mse(tTest - Y);
        
    end
    
    Acc = mean(tmpAcc);
    Err = mean(tmpErr);
    
end

end