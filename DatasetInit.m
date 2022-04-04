function [xTrain, tTrain, xTest, tTest] = DatasetInit(DatasetName)

% ===== FOR CLASSIFICATION DATASET ONLY ===== %

% Read dataset from files
% Generate Target Class
% Shuffle dataset's order
% Store in variables

% ==================================================
if DatasetName == "Cancer"
    %Load Data from File
    data = load('Cancer.txt');
    [X,~] = mapminmax(data(:,2:10),0,1);
    
    %Generate Target Class
    for i = 1:size(data,1)
        
        if data(i,11) == 2
            T(i,:) = [1 0];
        elseif data(i,11) == 4
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(699);
    xTrain = X(I(1:599),:);
    tTrain = T(I(1:599),:);
    xTest = X(I(591:end),:);
    tTest = T(I(591:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================
if DatasetName == "Heart"
    %Load Data from File
    data = load('Heart.txt');
    [X,~] = mapminmax(data(:,1:13),0,1);
    
    %Generate Target Class
    for i = 1:size(data,1)
        
        if data(i,14) == 0
            T(i,:) = [1 0];
        elseif data(i,14) == 1
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(303);
    xTrain = X(I(1:80),:);
    tTrain = T(I(1:80),:);
    xTest = X(I(81:end),:);
    tTest = T(I(81:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================
if DatasetName == "COVID"
    %Load Data from File
    data = load('covid-rael.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:8
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %/////////////////////
    
    %Generate Target Class
    for i = 1:size(data,1)
        
        if data(i,9) == 1
            T(i,:) = [1 0];
        elseif data(i,9) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(982);
        xTrain = X(I(1:737),:);
        tTrain = T(I(1:737),:);
        xTest = X(I(738:end),:);
        tTest = T(I(738:end),:);

    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================

if DatasetName == "COVID_UNDER"
    %Load Data from File
    data = load('under.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:8
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %/////////////////////
    
    %Generate Target Class
    for i = 1:size(data,1)
        
        if data(i,9) == 1
            T(i,:) = [1 0];
        elseif data(i,9) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(734);
    xTrain = X(I(1:587),:);
    tTrain = T(I(1:587),:);
    xTest = X(I(588:end),:);
    tTest = T(I(588:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================

if DatasetName == "COVID_OVER"
    %Load Data from File
    data = load('over.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:8
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %/////////////////////
    
    %Generate Target Class
    for i = 1:size(data,1)
        
        if data(i,9) == 1
            T(i,:) = [1 0];
        elseif data(i,9) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(1230);
    xTrain = X(I(1:1129),:);
    tTrain = T(I(1:1129),:);
    xTest = X(I(1130:end),:);
    tTest = T(I(1130:end),:);
    
    
    clear data I X T;% xTrain tTrain xTest tTest;
end
% ==================================================

% if DatasetName == "COVID22"
%     %Load Data from File
%     data = load('dataset2022.txt');
%     %[X,~] = mapminmax(data(:,2:10),0,1);
%     
%     for i = 2:8
%         [X(:,i-1),~] = mapminmax(data(:,i));
%     end
%     
%     %Generate Target Class
%     for i = 1:size(data,1)
%         
%         if data(i,9) == 1
%             T(i,:) = [1 0];
%         elseif data(i,9) == 2
%             T(i,:) = [0 1];
%         end
%         
%     end
%     
%     %Sampling Split Data
%     rng('default'); % Random seed
%     %rng(1); % Random seed
%     I = randperm(1513);
%     xTrain = X(I(1:1059),:);
%     tTrain = T(I(1:1059),:);
%     xTest = X(I(1060:end),:);
%     tTest = T(I(1060:end),:);
%     
%     clear data I X T;% xTrain tTrain xTest tTest;
% end

% ==================================================


if DatasetName == "COVID22"
    %Load Data from File
    data = load('covid2022.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:6
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %Generate Target Class
    for i = 1:size(data,1)
        if data(i,7) == 1
            T(i,:) = [1 0];
        elseif data(i,7) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(1513);
    xTrain = X(I(1:1059),:);
    tTrain = T(I(1:1059),:);
    xTest = X(I(1060:end),:);
    tTest = T(I(1060:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================
if DatasetName == "COVID_UNDER22"    
    %Load Data from File
    data = load('under2022.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:6
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %Generate Target Class
    for i = 1:size(data,1)
        if data(i,7) == 1
            T(i,:) = [1 0];
        elseif data(i,7) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(1286);
    xTrain = X(I(1:900),:);
    tTrain = T(I(1:900),:);
    xTest = X(I(901:end),:);
    tTest = T(I(901:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================
if DatasetName == "COVID_OVER22"
    %Load Data from File   
    data = load('over2022.txt');
    %[X,~] = mapminmax(data(:,2:10),0,1);
    
    for i = 2:6
        [X(:,i-1),~] = mapminmax(data(:,i));
    end
    
    %Generate Target Class
    for i = 1:size(data,1)
        if data(i,7) == 1
            T(i,:) = [1 0];
        elseif data(i,7) == 2
            T(i,:) = [0 1];
        end
        
    end
    
    %Sampling Split Data
    rng('default'); % Random seed
    %rng(1); % Random seed
    I = randperm(1740);
    xTrain = X(I(1:1218),:);
    tTrain = T(I(1:1218),:);
    xTest = X(I(1219:end),:);
    tTest = T(I(1219:end),:);
    
    clear data I X T;% xTrain tTrain xTest tTest;
end

% ==================================================

end