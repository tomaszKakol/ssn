clear all;clc;close all
%% przyk³ad 1 irys
%{ 
//tutaj jest problem z wytrenowaniem, nie wiem dlaczego
X_Setosa = irisInputs(:,1:50);
X_Virginica = irisInputs(:,51:100); 
X_Versicolour = irisInputs(:,101:150);
% wylosowanie 20 kolumn ka¿dego z gutunków do danych ucz¹cych
X_Setosa_train_i = randperm(50, 20); 
X_Virginica_train_i = randperm(50, 20) +50; 
X_Versicolour_i = randperm(50, 20) + 100; 
% zdefiniowanie wektora 60 danych testuj¹cych 3x20
X_train_iris_i = [X_Setosa_train_i  X_Virginica_train_i  X_Versicolour_i];
% wygenerowanie wektorów ucz¹cych i wektora wyjœcia
X_train = zeros(1:4, length(X_train_iris_i));
D_train = zeros(1:2, length(X_train_iris_i));
for  i = 1: length(X_train_iris_i)
    X_train(1:4, i) = irisInputs(:,X_train_iris_i(i)); 
    D_train(1:2, i) = irisTargets(1:2,X_train_iris_i(i)); 
end
% wygenerowanie wektorów ucz¹cych i wektora wyjœcia
X_test = zeros(1:4, length(irisInputs) - length(X_train_iris_i));  % inicjalizacja X_test
Y_test = zeros(1:2, length(irisTargets) - length(X_train_iris_i)); % inicjalizacja Y_test
A = [1:150];
check = zeros(1, length(irisInputs));
for i  = 1:length(A)
    check(i) = ismember(A(i),X_train_iris_i);
end

j=1;
for i  = 1:length(check)
    if(check(i) == 0)
     X_test(1:4, j) = irisInputs(:,i); 
     Y_test(1:2, j) =  irisTargets(1:2,i);
    j = j+1;
    end
end

% sieæ jako pojedynczy perceptron:
net = perceptron;
net.trainParam.epochs = 200;
net.trainParam.lr = 0.0001;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.002;
net = init(net);
net = train(net, X_train,D_train);
weights1 = net.IW{1,1}(1,:);
weights2 = net.IW{1,1}(2,:);
weights_value1 = sprintf('Wagi po³¹czeñ 1-wszego perceprtonu: %10.4f %10.4f  %10.4f  %10.4f ',weights1);
weights_value2 = sprintf('Wagi po³¹czeñ 2-ego perceprtonu: %10.4f %10.4f  %10.4f  %10.4f ',weights1);
disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
disp(weights_value1);
disp(weights_value2);
Y = net(X_test); 
error = length(find((Y ~= Y_test)))
q = error / length(Y)
error_value = sprintf('B³¹d: %10.4f  ',q);
disp('Wartoœæ wskaŸnika b³êdu na wyjœciu neuronu dla danych testuj¹cych:');
disp(error_value );
%}

% wczytanie danych IRIS
load iris_dataset;
d = 0.6; % tzw. test factor 

% podzia³ danych na ucz¹ce i testowe
const= size(irisTargets,2)/size(irisTargets,1);%50
test_i = 1;
uczenie_i = 1;
for i=1:size(irisTargets,1) %3kolumny
    for j=1:const
        if j<=d*const %d*50
            X_test(1:4,test_i) = irisInputs(1:4,j + (i-1)*const);
          if  irisTargets(2,j + (i-1)* const) == 1   %przypadek IrisVersicolour
                Y_test(1:2,test_i) = [1;1];
          elseif irisTargets(1,j + (i-1)* const) == 1           %przypadek IrisSetosa
                Y_test(1:2,test_i) = [1;0];
            else
                Y_test(1:2,test_i) = [0;1];                             %przypadek IrisVirginica
            end
            test_i = test_i + 1;
        else
            X_uczenie(1:4,uczenie_i) = irisInputs(1:4,j + (i-1)*const);
            if irisTargets(2,j + (i-1)*const) == 1        %przypadek IrisVersicolour
                D(1:2,uczenie_i) = [1;1];
            elseif irisTargets(1,j + (i-1)* const) == 1              %przypadek IrisSetosa
                D(1:2,uczenie_i) = [1;0];
            else
                D(1:2,uczenie_i) = [0;1];                                  %przypadek IrisVirginica
            end
            uczenie_i = uczenie_i + 1;
        end
    end
end

net = perceptron; % sieæ jako perceptron:
net.trainParam.epochs = 200;
net.trainParam.lr = 0.000001;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.02;
net = init(net);
net = train(net,X_uczenie,D);

weights1 = net.IW{1,1}(1,:);
weights2 = net.IW{1,1}(2,:);
weights_value1 = sprintf('Wagi po³¹czeñ 1-wszego perceprtonu: %10.4f %10.4f  %10.4f  %10.4f ',weights1);
weights_value2 = sprintf('Wagi po³¹czeñ 2-ego perceprtonu: %10.4f %10.4f  %10.4f  %10.4f ',weights1);
disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
disp(weights_value1);
disp(weights_value2);

Y = net(X_test);%zadanie na wyuczon¹ sieæ wartoœci testowych
error = length(find((Y ~= Y_test)));
q = error / length(Y);
error_value = sprintf('B³¹d: %10.4f   (iloœæ b³êdnie sklasyfikowanych danych: %10.0f )',q, error);% MSE danych testowych`
disp('Wartoœæ wskaŸnika b³êdu na wyjœciu neuronu dla danych testuj¹cych:');
disp(error_value );

%% przyk³ad 2  ionosphere
clear all; clc; close all
load ionosphere;
ionosphereInputs = X';
ionosphereTargets= [cell2mat(Y)=='g' cell2mat(Y)=='b']';
% przekszta³cenie danych, litery na cyfry 0 i 1
for iteration=1:9
    d = iteration/10; %wskaznik danych testowych do wszystkich danych
    
   const = size(ionosphereInputs, 1);
    sizeY = length(Y);
    test_i = 1;
    uczenie_i = 1;
    for i=1:sizeY
        if i<=d*sizeY           % dane testowe
            X_test(1:const,test_i) = ionosphereInputs(1:const,i);
            D_test(test_i) = ionosphereTargets(i);
            test_i = test_i + 1;
        else                             % dane ucz¹ce
            X_uczenie(1:const, uczenie_i) = ionosphereInputs(1:const,i);
            D_uczenie(uczenie_i) = ionosphereTargets(i);
            uczenie_i = uczenie_i + 1;
        end
    end
    
    net = perceptron;
    net.trainParam.epochs = 200;
    net.trainParam.lr = 0.00001;
    net.inputWeights{1,1}.initFcn = ('rands');
    net.trainParam.goal = 0.02;
    net = init(net);
    net = train(net,X_uczenie,D_uczenie);
   
    weights = net.IW{1,1}(1,:);
    weights_value = sprintf('Wagi po³¹czeñ  perceprtonu:%10.4f %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f %10.4f %10.4f  %10.4f  %10.4f ',weights);
    disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
    disp(weights_value);

    Y_test = net(X_test);                     % Dane testotwe na wejœcie wyuczonego perceptronu
    Y_uczenie = net(X_uczenie);     % Dane uczonce na wejœcie wyuczonego perceptronu

    blad_uczenie(iteration) = length(find((Y_uczenie ~= D_uczenie)));
    MSE_uczenie(iteration) = blad_uczenie(iteration) / length(Y_uczenie);
    blad_test(iteration) = length(find((Y_test ~= D_test)));
    MSE_test(iteration) = blad_test(iteration) / length(Y_test);
    blad_uczenie_value = sprintf('Dane ucz¹ce. B³¹d MSE: %10.4f   (iloœæ b³êdnie sklasyfikowanych danych: %4.0f  )', MSE_uczenie(iteration) , blad_uczenie(iteration));
    blad_test_value = sprintf('Dane testowe. B³¹d MSE: %10.4f   (iloœæ b³êdnie sklasyfikowanych danych: %4.0f  )',MSE_test(iteration), blad_test(iteration));
    info = sprintf('Wartoœæ wskaŸnika b³êdu na wyjœciu neuronu dla wskaŸnika podzia³u d = %1.1f  )',d);
    disp(info);
    disp(blad_uczenie_value);
    disp(blad_test_value); 
end
d_iteration = 0.1:0.1:0.9;
plot(d_iteration, MSE_test,'gx','linewidth',3);
hold on
plot(d_iteration, MSE_uczenie, 'bo','linewidth',3);
xlabel('WskaŸnik podzia³u [dane testowe / wszystkie dane]');
ylabel('MSE');
legend('Dane testowe','Dane ucz¹ce')
title('Charakterystyka b³êdu MSE w zale¿noœci od wartoœci wskaŸnika podzia³u danych');
%% przyk³ad 3 wine
clear all; close all; clc

load wine_dataset;
%https://www.kaggle.com/brynja/wineuci
%Liczba wyst¹pieñ ka¿dej klasy wina
valueSamplesClass1 = 59;
valueSamplesClass2 = 71;
valueSamplesClass3 = 48;

for iteration=1:9
    d = iteration/10;
    const = size(wineInputs, 1);
    Y1 = [1;0]; %Winnica 1
    Y2 = [0;1]; %Winnica 2
    Y3 = [1;1]; %Winnica 3
    uczenie_i = 1;
    test_i = 1;
    
    % 3 pêtle dla uczenia sieci ka¿dej z klas z t¹ sam¹ wartoœci¹ wskaŸnika d
    %1 klasa: valueSamplesClass1
    for i=1:valueSamplesClass1
        if i<=d*valueSamplesClass1       % dane testuj¹ce
            X_test(1:const ,test_i) = wineInputs(1:const ,i);
            if wineTargets(1,i) == 1
                D_test(1:2,test_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_test(1:2,test_i) = Y2;
            else
                D_test(1:2,test_i) = Y3;
            end
            test_i = test_i + 1;
        else                                                 % dane ucz¹ce
            X_uczenie(1:const ,uczenie_i) = wineInputs(1:const ,i);
            if wineTargets(1,i) == 1
                D_uczenie(1:2,uczenie_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_uczenie(1:2,uczenie_i) = Y2;
            else
                D_uczenie(1:2,uczenie_i) = Y3;
            end
            uczenie_i = uczenie_i + 1;
        end
    end
    %2 klasa: valueSamplesClass2
    for i=(1 + valueSamplesClass1):(valueSamplesClass1+valueSamplesClass2)%zakres danych 2 klasy: '1+ klasa1 ' do 'klasa1 + klasa2'
        if (i-valueSamplesClass1)<=d*valueSamplesClass2 % analogiczne, dane testuj¹ce 2 klasy
            X_test(1:const ,test_i) = wineInputs(1:const ,i);
            if wineTargets(1,i) == 1
                D_test(1:2,test_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_test(1:2,test_i) = Y2;
            else
                D_test(1:2,test_i) = Y3;
            end
            test_i = test_i + 1;
        else %  analogiczne, dane ucz¹ce 2 klasy
            X_uczenie(1:const ,uczenie_i) = wineInputs(1:const ,i); 
            if wineTargets(1,i) == 1
                D_uczenie(1:2,uczenie_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_uczenie(1:2,uczenie_i) = Y2;
            else
                D_uczenie(1:2,uczenie_i) = Y3;
            end
            uczenie_i = uczenie_i + 1;
        end
    end
     %3 klasa: valueSamplesClass3
     start = valueSamplesClass1+valueSamplesClass2+1;
done = valueSamplesClass1+valueSamplesClass2+valueSamplesClass3;
    for i=start :done % analogicznie, dla 3 klasy start od danych 'klasa1 + klasa2 + 1'
        if (i-valueSamplesClass1-valueSamplesClass2)<=d*valueSamplesClass3
            X_test(1:const ,test_i) = wineInputs(1:const ,i);
            if wineTargets(1,i) == 1
                D_test(1:2,test_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_test(1:2,test_i) = Y2;
            else
                D_test(1:2,test_i) = Y3;
            end
            test_i = test_i + 1;
        else
            X_uczenie(1:const ,uczenie_i) = wineInputs(1:const ,i);
            if wineTargets(1,i) == 1
                D_uczenie(1:2,uczenie_i) = Y1;
            elseif wineTargets(2,i) == 1
                D_uczenie(1:2,uczenie_i) = Y2;
            else
                D_uczenie(1:2,uczenie_i) = Y3;
            end
            uczenie_i = uczenie_i + 1;
        end
    end
    
    net = perceptron;
    net.trainParam.epochs = 200;
    net.trainParam.lr = 0.00001;
    net.inputWeights{1,1}.initFcn = ('rands');
    net.trainParam.goal = 0.02;
    net = init(net);
    net = train(net,X_uczenie,D_uczenie);

   weights1 = net.IW{1,1}(1,:);
   weights2 = net.IW{1,1}(2,:);
   weights_value1 = sprintf('Wagi po³¹czeñ  perceprtonu 1: %10.4f %10.4f %10.4f %10.4f  %10.4f  %10.4f  %10.4f %10.4f %10.4f %10.4f  %10.4f  %10.4f   %10.4f ',weights1);
   weights_value2 = sprintf('Wagi po³¹czeñ  perceprtonu 2: %10.4f %10.4f %10.4f %10.4f  %10.4f  %10.4f  %10.4f %10.4f %10.4f %10.4f  %10.4f  %10.4f   %10.4f ',weights2);    
   disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
   disp(weights_value1);
   disp(weights_value2);

    Ytest = net(X_test);             % Dane testowe na wejœcie wyuczonego perceptronu
    Ytrain = net(X_uczenie);    % Dane uczonce na wejœcie wyuczonego perceptronu
    Yall = net(wineInputs);
    blad_test(iteration) = length(find((Ytest ~= D_test))); 
    MSE_test(iteration) = blad_test(iteration) / length(Ytest);
    blad_uczenie(iteration) = length(find((Ytrain ~= D_uczenie)));
    MSE_uczenie(iteration) = blad_uczenie(iteration) / length(Ytrain);
    blad_uczenie_value = sprintf('Dane ucz¹ce. B³¹d MSE: %10.4f   (iloœæ b³êdnie sklasyfikowanych danych: %4.0f  )', MSE_uczenie(iteration) , blad_uczenie(iteration));
    blad_test_value = sprintf('Dane testowe. B³¹d MSE: %10.4f   (iloœæ b³êdnie sklasyfikowanych danych: %4.0f  )',MSE_test(iteration), blad_test(iteration));
    info = sprintf('Wartoœæ wskaŸnika b³êdu na wyjœciu neuronu dla wskaŸnika podzia³u d = %1.1f  )',d);
    disp(info);
    disp(blad_uczenie_value);
    disp(blad_test_value); 
end
d_iteration = 0.1:0.1:0.9;
plot(d_iteration, MSE_test,'gx','linewidth',3);
hold on;
plot(d_iteration, MSE_uczenie,  'bo','linewidth',3);
xlabel('WskaŸnik podzia³u [dane testowe / wszystkie dane]');
ylabel('MSE');
title('Charakterystyka b³êdu MSE w zale¿noœci od wartoœci wskaŸnika podzia³u danych');
legend('Dane testowe','Dane ucz¹ce')
grid on;
