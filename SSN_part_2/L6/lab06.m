%% 07.05.2018, Dane medyczne
clear all; close all; clc;
load Learning.mat
load Test.mat

%% SSN FEEDFORWARDNET
net1 = feedforwardnet(10);%[2 32 2] , 5 ...
net1.divideParam.trainRatio = 1;
net1.divideParam.valRatio   = 0;
net1.divideParam.testRatio  = 0;
net1.layers{1}.size = 3;
net1.layers{2}.size = 2;
net1.layers{1}.transferFcn = 'logsig';% logsig, radbas, tansig,
%net1.layers{2}.transferFcn = 'purelin';% logsig, radbas, tansig,
net1.inputWeights{1,1}.initFcn = ('rands');
net1.trainParam.goal = 0.0001;
%net1.trainFcn = 'traingd'
net1.trainParam.epochs = 200;
net1.trainParam.lr = 0.0001;
net1 = configure(net1, Xlearning, Dlearning);
net1 = train(net1, Xlearning, Dlearning);
view(net1)
Ylearning = net1(Xlearning);
Ytest = net1(Xtest);

%Dlearning, Dtest {-1, 1} -> Ylearning, Ytest {-1, 1}
for i=1:31
    if Ylearning(i) > 0
        Ylearning_round(i) = 1;
    else
        Ylearning_round(i) = -1;
    end
    if Ytest(i) > 0
        Ytest_round(i) = 1;
    else
        Ytest_round(i) = -1;
    end
end

error_learning = length(find((Ylearning_round ~= Dlearning)));
error_learning_printf = sprintf('SSN: Liczba b³êdnie sklasyfikowanych danych ucz¹cych Dlearning: %3.0f  ',error_learning);
disp(error_learning_printf);
MSE_learning = mse(net1,Dlearning,Ylearning);
MSE_learning_printf = sprintf('SSN: MSE dla uczenia Dlearning, d1: %10.4f  ',MSE_learning);
disp(MSE_learning_printf);

error_test = length(find((Ytest_round ~= Dtest)));
error_test_printf = sprintf('SSN: Liczba b³êdnie sklasyfikowanych danych testuj¹cych Dtest: %3.0f  ',error_test);
disp(error_test_printf);
MSE_test = mse(net1,Dtest,Ytest);
MSE_test_printf = sprintf('SSN: MSE dla testowania Dtest, d1: %10.4f  ',MSE_test);
disp(MSE_test_printf);

%% RBF 

spread = 8.7;
K = 400;
goal = 0;
Ki = 1;
net2 = newrb( Xlearning, Dlearning,goal,spread,K,Ki);
view (net2)

Y_test_rbf = net2(Xtest);
Y_learning_rbf = net2(Xlearning);

%Dlearning, Dtest {-1, 1} -> Ylearning, Ytest {-1, 1}
for i=1:31
    if Y_learning_rbf (i) > 0
        Ylearning_round_rbf(i) = 1;
    else
        Ylearning_round_rbf(i) = -1;
    end
    if Y_test_rbf(i) > 0
        Ytest_round_rbf(i) = 1;
    else
        Ytest_round_rbf(i) = -1;
    end
end

error_learning_rbf = length(find((Ylearning_round_rbf~= Dlearning)));
error_learning_printf_rbf = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych ucz¹cych Dlearning: %3.0f  ',error_learning_rbf);
disp(error_learning_printf_rbf);
MSE_learning_rbf = mse(net2,Dlearning,Y_learning_rbf);
MSE_learning_printf_rbf = sprintf('RBF: MSE dla uczenia Dlearning, d1: %10.4f  ',MSE_learning_rbf);
disp(MSE_learning_printf_rbf);

error_test_rbf = length(find((Ytest_round_rbf~= Dtest)));
error_test_printf_rbf = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych testuj¹cych Dtest: %3.0f  ',error_test_rbf);
disp(error_test_printf_rbf);
MSE_test_rbf = mse(net2,Dtest,Y_test_rbf );
MSE_test_printf_rbf = sprintf('RBF: MSE dla testowania Dtest, d1: %10.4f  ',MSE_test_rbf);
disp(MSE_test_printf_rbf);


%% GRNN
startBestError_test = 10e5;
%for spread = 1:0.05:6
spread = 4.92;
net3 = newgrnn( Xlearning, Dlearning, spread);
view (net3)
Y_test_grnn = net3(Xtest);
Y_learning_grnn = net3(Xlearning);

for i=1:31
    if Y_learning_grnn (i) > 0
        Ylearning_round_grnn(i) = 1;
    else
        Ylearning_round_grnn(i) = -1;
    end
    if Y_test_grnn(i) > 0
        Ytest_round_grnn(i) = 1;
    else
        Ytest_round_grnn(i) = -1;
    end
end

error_learning_grnn = length(find((Ylearning_round_grnn~= Dlearning)));
error_learning_printf_grnn = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych ucz¹cych Dlearning: %3.0f  ',error_learning_grnn);
disp(error_learning_printf_grnn);
MSE_learning_grnn = mse(net3,Dlearning,Y_learning_grnn);
MSE_learning_printf_grnn = sprintf('RBF: MSE dla uczenia Dlearning, d1: %10.4f  ',MSE_learning_grnn);
disp(MSE_learning_printf_grnn);

error_test_grnn = length(find((Ytest_round_grnn~= Dtest)));
error_test_printf_grnn = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych testuj¹cych Dtest: %3.0f  ',error_test_grnn);
disp(error_test_printf_grnn);
MSE_test_grnn = mse(net3,Dtest,Y_test_grnn );
MSE_test_printf_grnn = sprintf('RBF: MSE dla testowania Dtest, d1: %10.4f  ',MSE_test_grnn);
disp(MSE_test_printf_grnn);

if MSE_test_grnn <startBestError_test
    startBestError_test = MSE_test_grnn ;
     bestNetwork_test = net3;
    bestSpread_test = spread;
end
%end
%disp(sprintf('Najlepsze dopasowanie:'));
%disp(sprintf('B³¹d MSE danych testuj¹cych: %1.10f  , wspó³czynnik Spread: %1.2f  ', startBestError_test , bestSpread_test ));

%%PNN
 Dlearning =Dlearning + 2;
 Dtest =Dtest +2;
T = ind2vec(Dlearning);


startBestError_test = 10e5;
%for spread = 2.5:0.05:4
spread = 3;
net4 = newpnn( Xlearning, T, spread);
view (net4)
Y_test_pnn = net4(Xtest);
Y_test_pnn =vec2ind(Y_test_pnn );
Y_learning_pnn = net4(Xlearning);
Y_learning_pnn =vec2ind(Y_learning_pnn);

for i=1:31
    if Y_learning_pnn (i) > 2
        Ylearning_round_pnn(i) = 3;
    else
        Ylearning_round_pnn(i) = 1;
    end
    if Y_test_pnn(i) > 2
        Ytest_round_pnn(i) = 3;
    else
        Ytest_round_pnn(i) = 1;
    end
end

error_learning_pnn = length(find((Ylearning_round_pnn~= Dlearning)));
error_learning_printf_pnn = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych ucz¹cych Dlearning: %3.0f  ',error_learning_pnn);
disp(error_learning_printf_pnn);
MSE_learning_pnn = mse(net4,Dlearning,Y_learning_pnn);
MSE_learning_printf_pnn = sprintf('RBF: MSE dla uczenia Dlearning, d1: %10.4f  ',MSE_learning_pnn);
disp(MSE_learning_printf_pnn);

error_test_pnn = length(find((Ytest_round_pnn~= Dtest)));
error_test_printf_pnn = sprintf('RBF: Liczba b³êdnie sklasyfikowanych danych testuj¹cych Dtest: %3.0f  ',error_test_pnn);
disp(error_test_printf_pnn);
MSE_test_pnn = mse(net4,Dtest,Y_test_pnn );
MSE_test_printf_pnn = sprintf('RBF: MSE dla testowania Dtest, d1: %10.4f  ',MSE_test_pnn);
disp(MSE_test_printf_pnn);

if MSE_test_pnn <startBestError_test
    startBestError_test = MSE_test_pnn ;
     bestNetwork_test = net4;
    bestSpread_test = spread;
end
%end
%disp(sprintf('Najlepsze dopasowanie:'));
%disp(sprintf('B³¹d MSE danych testuj¹cych: %1.10f  , wspó³czynnik Spread: %1.2f  ', startBestError_test , bestSpread_test ));
