%save('C:\Users\tomek\Desktop\I_sem\SSN\lab_04_05_06\SPRAWKO\L5/wine.mat');
clear all; close all; clc;

data = {'IRIS', 'WINE', 'IONO'};

vector = 0.1:0.1:0.9;
startBestError_test = 1e5;
startBestError_train = 1e5;
MSE_matrixTest = zeros(3, 20 , size(vector,2));
MSE_matrixTrain = zeros(3, 20, size(vector,2));
errorTrain= zeros(3, 20, size(vector,2));
 errorTest = zeros(3, 20, size(vector,2));
   
data={'IRIS','WINE','IONO'};
setData=3;
if strcmp(data{setData}, 'IRIS')
   load iris_dataset;
   X_inputs = irisInputs;
   T_outputs = irisTargets;
elseif strcmp(data{setData}, 'WINE')
   load wine_dataset;
   X_inputs = wineInputs;
   T_outputs = wineTargets;
elseif strcmp(data{setData}, 'IONO')
   load ionosphere;
   X_inputs = X';
   T_outputs = [cell2mat(Y) == 'g' cell2mat(Y) == 'b']';
end
nntraintool('close')
[sizeDataY, sizeDataX] = size(X_inputs);
outputs = zeros(1, sizeDataX);

for i=1:length(T_outputs)
    [M1,I] = max(T_outputs(:, i));
    outputs(i) = I;
end

 figure(setData)
 
for D =  0.1:0.1:0.9

   for ITERATION = 1:20
     % sta³e
    randPerm = randperm(sizeDataX);
    const_1 = D * sizeDataX;
    const_2 = (1 - D) * sizeDataX;
    % inicjalizacja 
    X_train = zeros(sizeDataY, floor(const_1));
    X_test = zeros(sizeDataY, floor(const_2 ));
    T_train = zeros(1, floor(const_1));
    T_test = zeros(1, floor(const_2 ));
        Y_train_round= zeros(1, floor(const_1));
    Y_test_round = zeros(1, floor(const_2 ));
    for i = 1:sizeDataX
        if i <= floor(const_1)
            X_train(:, i) = X_inputs(:, randPerm(i));
            T_train(i) = outputs(randPerm(i))-1;
        else
            X_test(:, i - floor(const_1)) = X_inputs(:, randPerm(i));
            T_test(i - floor(const_1)) = outputs(randPerm(i))-1;
        end
    end
    M = zeros(sizeDataY, 2);
    for i = 1:sizeDataY
        M(i, 1) = min(X_inputs(i,:));
        M(i, 2) = max(X_inputs(i,:));
    end

    goal = 0.001;
    net = feedforwardnet(7);
    net.layers{1}.size = 34;
    net.layers{2}.size = 2;
    net.layers{1}.transferFcn = 'logsig';%logsig
   % net.layers{2}.transferFcn = 'tansig';%purelin
    net.inputWeights{1,1}.initFcn = ('rands');
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio   = 0;
    net.divideParam.testRatio  = 0;
    net.trainParam.goal = goal;
    net.trainParam.epochs = 200;
    net.trainParam.lr =  0.00001;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net = configure(net, X_train, T_train);
    net = train(net, X_train, T_train);
   
    Y_test = sim(net, X_test);
    Y_train = sim(net, X_train);
   
    for i=1: size(T_train,2)
        if Y_train(i) > 0.5
            Y_train_round(i) = 1;
        else
             Y_train_round(i) = 0;
        end
    end
    for i = 1:size(T_test,2)
        if Y_test(i) > 0.5
            Y_test_round(i) = 1;
        else
             Y_test_round(i) = 0;
        end
     end


    J_index =  find(vector<=D);
     j = ind2sub(size(vector), J_index(1, size(J_index ,2)));
     MSE_test = perform(net, T_test, Y_test);
     MSE_train = perform(net, T_train, Y_train );
      MSE_matrixTest(setData, ITERATION ,j) = MSE_test;
    MSE_matrixTrain(setData, ITERATION, j) = MSE_train;
     %MSE_test_printf = sprintf('MSE dla testowania: %10.5f  ',MSE_test);
     %disp(MSE_test_printf);
     %MSE_train_printf = sprintf('MSE dla uczenia: %10.5f  ',MSE_train );
     %disp(MSE_train_printf);
       J_index =  find(vector<=D);
     j = ind2sub(size(vector), J_index(1, size(J_index ,2)));
  averageMSE_Test_Print = mean(mean(MSE_matrixTest(setData, :, j),1),2);
  averageMSE_Train_Print = mean(mean(MSE_matrixTrain(setData, :, j),1),2);
    
      errorTrain(setData, ITERATION ,j) = length(find((Y_train_round ~= T_train)));
     errorTest(setData, ITERATION ,j) = length(find((Y_test_round ~= T_test)));
    averageTestERROR =mean(mean( errorTest(setData, :, j),1),2);
   averageTrainERROR =mean(mean( errorTrain(setData, :, j),1),2);
   
   ratioErrorTest = 100 *(1-D)*(size(T_outputs,2) - averageTestERROR)/((1-D)*size(T_outputs,2));
    ratioErrorTrain= 100 *(D)*(size(T_outputs,2) - averageTrainERROR)/((D)*size(T_outputs,2));  
     
   if MSE_train<startBestError_train
                startBestError_train = MSE_train;
                bestNetwork_train = net;
                bestD_train = D;
                 bestErrorTrain=min(errorTrain(setData, :, j),1);
   end
     if MSE_test<startBestError_test
        startBestError_test = MSE_test;
        bestNetwork_test = net;
        bestD_test = D;
        best_Iteration = ITERATION;
        bestErrorTest=min(errorTest(setData, :, j),1);
     end
    
  end
 
disp('Wyniki z testów jednostkowych:')
disp(sprintf('Dane nr: %1.0f , wspó³czynnik D: %1.2f  ', setData, D));
disp(sprintf('B³¹d œredni MSE Test: %1.10f ',averageMSE_Test_Print  ));
disp(sprintf('B³¹d œredni MSE Train: %1.10f ',averageMSE_Train_Print ));
disp(sprintf('SSN: Œrednia b³êdnie sklasyfikowanych danych ucz¹cych: %3.0f  ',averageTrainERROR ));
disp( sprintf('SSN: Œrednia  b³êdnie sklasyfikowanych danych testuj¹cych: %3.0f  ',averageTestERROR));
disp(sprintf('SSN: Procenowy wskaŸnik b³êdu danych ucz¹cych: %3.2f  % ',ratioErrorTrain));
disp( sprintf('SSN: Procenowy wskaŸnik b³êdu danych testuj¹cych: %3.2f  % ',ratioErrorTest ));
  
hold on
  subplot(2,1,1)
plot(D,averageMSE_Train_Print,'rx','Linewidth',5);
xlabel('Wspó³czynnik D')
ylabel('MSE train')
 xlim([0, vector(size(vector,2))+0.04])
title(sprintf('Numer danych: %1.0f  Dane ucz¹ce. \n Charakterystyka zmiany œredniego  b³êdu MSE od wartoœci wspó³czynnika D. ', setData ))

hold on;
subplot(2,1,2) 
plot(D,averageMSE_Test_Print,'rx','Linewidth',5);
xlabel('Wspó³czynnik D')
ylabel('MSE test')
 xlim([0, vector(size(vector,2))+0.04])
title(sprintf('Numer danych: %1.0f  Dane testuj¹ce. \n Charakterystyka zmiany œredniego  b³êdu MSE od wartoœci wspó³czynnika D. ',setData ));


end

disp(sprintf('Najlepsze dopasowanie:'));
disp(sprintf('B³¹d MSE danych testuj¹cych: %1.10f , wspó³czynnik D: %1.2f, iteracja: %2.0f, B³êdnych dopasowañ: %3.1f ', startBestError_test , bestD_test ,best_Iteration ,bestErrorTest));
disp(sprintf('B³¹d MSE danych ucz¹cych: %1.10f , wspó³czynnik D: %1.2f , iteracja: %2.0f , B³êdnych dopasowañ: %3.1f', startBestError_train , bestD_train,best_Iteration,bestErrorTrain ));
view(net)


%end

                



