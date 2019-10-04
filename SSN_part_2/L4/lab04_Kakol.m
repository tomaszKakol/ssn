%% Tomasz K¹kol, lab04 SSN 2018
clear all; close all; clc;

%% funkcja
A = 1; B =2; C = 30; D = 8; E = -60; 
func = @(x) A*sin(B*x+C).*cos(D*x+E);
a_train = 5; b_test = 5;
step_train = 0.01; step_test = 0.001;

%% tworzenie zbioru ucz¹cego
x_train = 0:step_train:a_train;
n_train = size(x_train,2);
y_train = zeros(1,n_train);
for i=1:n_train
    y_train(i) = func(x_train(i));
end
%plot(x_train,y_train);

%% tworzenie zbioru testowego
x_test = 0:step_test:b_test;
n_test = size(x_test,2);
y_test = zeros(1,n_test);
for i=1:n_test
    y_test(i) = func(x_test(i));
end
%figure 
%plot(x_test,y_test);

%% inicjalizacja sieci
N = 15; % max liczba neuronów
neuronsNum = zeros(1,N);
MSE = zeros(1,N);
nntraintool('close')
startBestError = 1e5;
startBestError_test =  1e5;
activationFcn = {'tansig'};%'logsig','tansig','purelin'
maxEpochs= 200;
stepEpochs= 5;
LR = 0.000001;

t_train = zeros(size(activationFcn, 2), N);
Epochs_iteration = 1 : stepEpochs: (stepEpochs*(floor(maxEpochs/stepEpochs) -1)+1);
for z  = 1:size(activationFcn, 2)
       Rr_test = zeros(N, maxEpochs);
       MSE = zeros(N, maxEpochs);
       MSE_train_tabel = zeros(N, maxEpochs);
        for i=1:N
            display(sprintf('Przypadek: liczba neuronów: %2.0f  ', i));
             for k = 1:stepEpochs:maxEpochs
                % display(sprintf('Wartoœæi konfiguracyjne:  liczba epok: %3.0f ,  f. aktywacji: %s  ', k, activationFcn{z}));
                neurons = i; % liczba neuronów
                net = feedforwardnet(neurons);
                net.divideParam.trainRatio = 1;
                net.divideParam.valRatio   = 0;
                net.divideParam.testRatio  = 0;
                net.inputWeights{:,:}.initFcn = ('rands');
                net1.layers{1}.size = 15;
                net.layers{1}.transferFcn = activationFcn{z};

                net = configure(net,x_train,y_train);
                net.trainParam.epochs=maxEpochs;
                net1.trainParam.lr = LR;
                net.trainParam.showWindow = false;
                net.trainParam.showCommandLine = false;
                tic % zegar start
                net = train(net,x_train,y_train);
                t_train(z,i)= toc; % zegar stop
               % disp(sprintf('- czas trenowania sieci: %10.3f  [s]',t_train(z,i,k)));

                Y = net(x_test);
                Y_train = net(x_train);
                MSE_train = perform(net, y_train, Y_train );
                MSE_train_tabel(i,k) = MSE_train;
               % MSE_train_printf = sprintf('- MSE dla uczenia: %10.5f  ',MSE_train );
               % disp(MSE_train_printf);
                MSE_test = perform(net, y_test, Y);
                %MSE_test_printf = sprintf('- MSE dla testowania: %10.5f  ',MSE_test);
                %disp(MSE_test_printf);
                MSE(i,k) = MSE_test;

                AA_test = [y_test' Y'];
                R_test = corrcoef(AA_test);
                Rr_test(i, k) = R_test(1,2) ;

                AA_train = [y_train' Y_train'];
                R_train = corrcoef(AA_train);
                Rr_train(i, k) = R_train(1,2) ;

                if MSE_test<startBestError_test
                    startBestError_test = MSE_test;
                    bestNetwork_test = net;
                    bestTransferFcn_test = activationFcn{z};
                    bestNeurons_test = i;
                    bestEpochs_test = k;
                end
                
                if MSE_train<startBestError
                    startBestError = MSE_train;
                    bestNetwork = net;
                    bestTransferFcn = activationFcn{z};
                    bestNeurons = i;
                    bestEpochs = k;
                end
               %  disp(' -- ')
             end
             if(i==1 || i==5|| i==10 || i==15)
                 figure
                 plot(y_test, y_test ,'bo')
                 hold on
                 plot(y_test, Y ,'ro')
                 xlabel('y test ')
                 ylabel('y test,  Y')
                 title(sprintf('Charakterystyka R danych testowych, l. Neuronów: %2.0f, l. Epok: 196, f.aktywacji: %s ',i,  activationFcn{z}))
                 legend('y test &y test','y test & Y')

                figure
                plot(Epochs_iteration,Rr_train(i,Epochs_iteration),'gx','Linewidth',4);
                xlabel('Epoki')
                ylabel('R')
                title(sprintf('Dane ucz¹ce: f. aktywacji: %s , , liczba neuronów %2.0f', activationFcn{z}, i)) %f. treningu: %s  trainFcn{j}

                 figure
                plot(Epochs_iteration,Rr_test(i,Epochs_iteration),'gx','Linewidth',4);
                xlabel('Epoki')
                ylabel('R')
                title(sprintf('Dane, testuj¹ce: f. aktywacji: %s , , liczba neuronów %2.0f', activationFcn{z}, i)) %f. treningu: %s  trainFcn{j}

                figure
                plot(Epochs_iteration,MSE(i,Epochs_iteration),'rx','Linewidth',4);
                xlabel('Epoki')
                ylabel('MSE')
                title(sprintf('Dane testuj¹ce: f. aktywacji: %s ,  liczba neuronów %2.0f', activationFcn{z}, i))%f. treningu: %s ,, trainFcn{j}

                figure
                plot(Epochs_iteration,MSE_train_tabel(i,Epochs_iteration),'rx','Linewidth',4);
                xlabel('Epoki')
                ylabel('MSE test')
                title(sprintf('Dane ucz¹ce: f. aktywacji: %s ,  liczba neuronów %2.0f', activationFcn{z}, i))%f. treningu: %s ,, trainFcn{j}

                 figure 
                 plot(x_train,y_train,'b');
                 hold on
                 plot(x_test,Y,'r');
                 xlabel('t')
                 ylabel('Amplituda')
                 legend('x train & y train','x test & Y')
                 title(sprintf('Dane ucz¹ce: f. aktywacji: %s ,  liczba neuronów %2.0f', activationFcn{z}, i))%f. treningu: %s ,, trainFcn{j}
             end
        end
end
disp(sprintf('Najlepsze dopasowanie:'));
disp(sprintf('B³¹d MSE danych testuj¹cych: %1.10f , liczba neuronów: %2.0f ,  f. aktywacji: %s , l. epok %3.0f  ', startBestError , bestNeurons , bestTransferFcn, bestEpochs ));
disp(sprintf('B³¹d MSE danych ucz¹cych: %1.10f , liczba neuronów: %2.0f ,  f. aktywacji: %s , l. epok %3.0f  ', startBestError_test , bestNeurons_test , bestTransferFcn_test, bestEpochs_test ));


