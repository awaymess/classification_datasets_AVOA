%%  Training Feed-forward Neural Networks using Optimizer %%

clear all;
clc;

%% Dataset No. %%
% classification datasets %
DatasetName = {'Cancer';'Heart';'COVID';'COVID_UNDER';'COVID_OVER';'COVID22';'COVID_UNDER22';'COVID_OVER22'};
OptimizerName = {'GWO';'AVOA';'GBO'};

%% Parameters Configuration %%
 
RunNo = 2;                                   % Max Run
SearchAgentsNo=30;                         % Number of search agents
MaxIteration = 100;                        % Maximum number of iterations / SearchAgents
Findex='null';                             % Function Index

for DatasetNo = 7
    
    % Dataset
    CurrentDataset = string(DatasetName(DatasetNo));
    disp(strcat('Working on ',CurrentDataset,' Dataset'));
    %fprintf('\n');
    
    for OpimizerNo = 1:size(OptimizerName,1)
        
        % Optimizer
        CurrentOptimizer = string(OptimizerName(OpimizerNo));
        disp(strcat(string(OptimizerName(OpimizerNo)),' is Running'));
        
        % Change Number of Hidden Node
        for HiddenNode = 22

            % Load details of the selected dataset.
            [lb,ub,dim,fobj,inp,hidn,outp] = GetFunctionsInfo(['F' num2str(DatasetNo)],HiddenNode);
            
            % Parameters for MLP
            mlpConfig.inp = inp;
            mlpConfig.hidn = hidn;
            mlpConfig.outp = outp;
            
            parfor run = 1:RunNo
                
                watchRun = tic; % Elapsed time for each run.
                
                if OpimizerNo == 1
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurveGWO(run,:)] = GWO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 2
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurveAVOA(run,:)] = AVOA(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end
                if OpimizerNo == 3
                    [BestScore(run,:),BestPosition(run,:),ConvergenceCurveGBO(run,:)] = GBO(SearchAgentsNo,MaxIteration,lb,ub,dim,fobj,mlpConfig);
                end

                elapsedRun = toc(watchRun);
                ElapsedTimeRun(run,:) = elapsedRun;     % Elapsed time for each run.
                disp(strcat(' >>  ',CurrentOptimizer,' Run No.',num2str(run),' --> ',num2str(HiddenNode),' Hidden Node is done. (',num2str(elapsedRun),' s.)'));
                
            end
 
            OutputDir = strcat('Results\latest');
            if ~exist(OutputDir, 'dir')
                mkdir(OutputDir);
            end
            
%           Save to file.
            filename = strcat('Results\latest\',CurrentDataset,'_',CurrentOptimizer,'_',num2str(HiddenNode),'_HiddenNode','_Weight_DATA.mat');
            save(filename,'BestPosition','BestScore');
            
            filename = strcat('Results\latest\',CurrentDataset,'_',CurrentOptimizer,'_',num2str(HiddenNode),'_HiddenNode','_ElapsedTime_DATA.mat');
            save(filename,'ElapsedTimeRun');
     
            % Testing rate
            [ClassificationRate(OpimizerNo,HiddenNode), ApproximationError(OpimizerNo,HiddenNode)] = TestFitness(['F' num2str(DatasetNo)],RunNo,mlpConfig,BestPosition);
            
             clear BestPosition BestScore ConvergenceCurve ElapsedTimeRun ;
            
        end
        
    end
    
%   Save to file.
    filename = strcat('Results\latest\',CurrentDataset,'_Performance_Summary_DATA.mat');
    save(filename,'ClassificationRate', 'ApproximationError','ConvergenceCurveGWO','ConvergenceCurveAVOA','ConvergenceCurveGBO');
    
%   clear ClassificationRate ApproximationError;
    
    disp(['Dataset ' num2str(DatasetNo) ' Finished']);
    fprintf('\n');
    
end
    
 display('--------------------------------------------------------------------------------------------')
 display('Classification rate')
 display('    MLP_GWO    MLP_AVOA     MLP_GBO ')
 display(mean(ClassificationRate(:,(HiddenNode)),2))
 display('--------------------------------------------------------------------------------------------')
 figure('Position',[500 500 660 290])
% Draw convergence curves

subplot(1,2,1);
hold on
title('Convergence Curves')
semilogy(mean(ConvergenceCurveGWO,1),'k')
semilogy(mean(ConvergenceCurveAVOA,1),'g')
semilogy(mean(ConvergenceCurveGBO,1),'r')

xlabel('Generation');
ylabel('MSE');

axis tight
grid on
box on
legend('GWO','AVOA','GBO')

% Draw classification rates
subplot(1,2,2);
hold on
title('Classification Accuracies')
bar(mean(ClassificationRate(:,(HiddenNode)),2))
xlabel('Algorithm');
ylabel('Classification rate (%)');

grid on
box on
set(gca,'XTickLabel',{'GWO','AVOA','GBO'});

