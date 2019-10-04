%% Sztuczne sieci neuronowe - laboratorium 2
%
% # Autor: Tomasz K¹kol
% # Data: 12.03.2018 - 26.03.2018
% # AGH
% # Kierunek: Informatyka Stosowana
% # Wydzia³: Fizyki i Informatyki Stosowanej
% 
% 

%% Wstêp
% 
% Podczas drugich zajêæ laboratoryjnych zadaniem by³o stworzenie  neuronu typu liniowego, 
% posiadaj¹cego umiejêtnoœæ wykonywania prostych obliczeñ dodawania i mno¿enia;
% stosuj¹c  2 wejœcia neuronu, uwzglêdniaj¹c oczekiwanych wartoœci wag po³¹czeñ.
%

%% Dodawanie
%
% Równanie postaci: 
%
% * 3*x_1 + 2*x_2  = Theta 
%
% Oczekiwanym przez nas rezultatem nauczania neuronu jest uzyskanie wag o wartoœci 3 i 2 dla kolejno  wejœcia x_1 oraz x_2.
% W celu przygotowania prostego widoku graficznego,  pocz¹tkowe wartoœci wektorów wejœæ  x_1 oraz x_2 s¹ równe, zawieraj¹c siê w  przedziale (0, 10), z przrostem równym 0.1.
% Nastêpnie, analogicznie jak podczas laboratorium 1, dane ucz¹ce zaszumiono z wykorzystaniem znormalizowanej funkcji randn. 
% Wartoœæ odchylenia standardowego i ampitudy dla zastosowanego szumu to odpowiednio 0.1 i 1.
% Dla przypadku danych testuj¹cych zakres wartoœci wartoœci ustawiono taki
% sam, ale zmniejszy³em próbê danych testuj¹ych, poprzez zwiêkszenie wartoœci przyrostu wektorów testuj¹ych.
% Dla zastosowanego typu sieci, powszechnym wskaŸnikiem wielkoœci b³êdu jest b³¹d œredniokwadratowy(ang. MSE, Mean squared error).
%
% Wyniki przeprowadzonego uczenia sieci:
clear all; close all; clc;

sigma = 0.1;    % Odchylenie standardowe
x1_1 = 0;   x1_2 = 10;    % krañce wektora wejœæ 1
x2_1 = 0;   x2_2 = 10;    % krañce wektora wejœæ 2
dx_1 = .1;  dx_2 = .1;    % krok wektorów wejœæ

% Wygenerowanie danych ucz¹cych, X_train oraz T_train
n = 1;
for i=x1_1:dx_1:x1_2                   %iteracja po ca³ym wektorze wejœæ 1
    for j=x2_1:dx_2:x2_2               %iteracja po ca³ym wektorze wejœæ 2
         X_train(1,n) = i + sigma*randn(1,1);
         X_train(2,n) = j + sigma*randn(1,1);
         D_train(1,n) = 3 * X_train(1, n) + 2 * X_train(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

% Przyrost wartoœci wektora testuj¹cego nuron
dx_3 = 0.8; dx_4 = 0.8;

% Wygenerowanie danychtestuj¹cych, X_test oraz T_test
n = 1;
for i = x1_1:dx_3:x1_2                   
    for j = x2_1:dx_4:x2_2               
         X_test(1,n) = i + sigma*randn(1,1);
         X_test(2,n) = j + sigma*randn(1,1);
         D_test(1,n) = 3 * X_test(1, n) + 2 * X_test(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

learningRate = 0.0000001;    % wspó³czynnik szybkoœci uczenia         
net = newlin(X_train, D_train, [0], learningRate);  % konfiguracja sieæ (newlin - neuron liniowy)
net.trainParam.epochs = 1000;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.0000001;
net = init(net); %inicjalizacja sieci
init_W = net.IW{1,1};    
init_W_value = sprintf('Wagi: %10.4f %10.4f ',init_W);
disp('Pocz¹tkowe wartoœci wag po³¹czeñ:');
disp(init_W_value);

% Uczenie sieci danymi ucz¹cymi
net = train(net, X_train, D_train);
weights = net.IW{:,1};
weights_value = sprintf('Wagi: %10.4f %10.4f ',weights);
disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
disp(weights_value);
% view(net); % Schemat zastosowanego neuronu typu liniowego
%%
% 
% <<lab02_perceptron.PNG>>
% 
% Obliczenie b³êdu metod¹ MSE dla danych ucz¹cych
Y_train = net(X_train);
MSE_train = mse(net, D_train, Y_train);
MSE_train_error = sprintf('MSE: %10.4f  ',MSE_train);
disp('Wartoœæ b³êdu danych ucz¹cych na wyjœciu neuronu, obliczonego metod¹ najmniejszych kwadratów wynosi:');
disp(MSE_train_error);

Y = net(X_test); % Podanie na wejœcie nuronu danych testuj¹cych
MSE_test = mse(net, D_test, Y); %Mean-squared error
MSE_test_error = sprintf('MSE: %10.4f  ',MSE_test);
disp('Wartoœæ b³êdu danych testuj¹ych na wyjœciu neuronu, obliczonego metod¹ najmniejszych kwadratów wynosi:');
disp(MSE_test_error );

% Edycja wymiarów wektorów wejœciowych i wyjœciowego do poprawnego
% wyœwietlenia danych
s = sqrt(length(X_test));
X_test_1 = reshape(X_test(1,:),s,s);
X_test_2 = reshape(X_test(2,:),s,s);
Y_1 = reshape(Y,s,s);

% Wyœwietlenie wyników dzia³ania sieci dla danych testowych
figure(1)
surf(X_test_1,X_test_2, Y_1); % lub mesh
xlabel('Wejœcie 1');
ylabel('Wejœcie 2');
zlabel('Wyjœcie neuronu');
title('Wykres wartoœci na wyjœciu nueronu, gdy na jego wejœcia wprowadzono dane testowe.');
grid on

%Wyznaczenie wektora b³êdu na wyjœciu neuronu
D_1=reshape(D_test,s,s);
error = Y_1-D_1;

figure(2)
surf(error)
title('Wykres b³êdu, jako ró¿nicy wartoœci oczekiwanych  i testowych wektora wyjœciowego.');
xlabel('Wejœcie 1');
ylabel('Wejœcie 2');
zlabel('Amplituda b³êdu na wyjœciu neuronu');
grid on

%%
%
%%
% Podczas wykonanych testów, widoczna jest poprawa minimalizacji wielkoœci
% b³êdu œredniokwadratowego wraz z zwêkszon¹ liczb¹ epok podczas
% testowania. Du¿¹ zmiennoœæ wyników uzyskuje siê poprzez edycjê parametru learning rate -  
% wydajnoœæ algorytmu jest bardzo wra¿liwa na w³aœciwe ustawienie szybkoœci uczenia siê. 
% Jeœli tempo uczenia siê posiada wartoœæ zbyt du¿¹, algorytm mo¿e oscylowaæ i staæ siê niestabilny. 
% Jeœli szybkoœæ uczenia siê jest zbyt ma³a, algorytm potrzebuje zbyt wiele czasu (epok), aby prawid³owo pogrupowaæ dane podawane na wejœcia.
% Ustalenie optymalnej wartoœci szybkoœci uczenia siê przed treningiem nie jest  w praktyce mo¿liwe. 
% W rzeczywistoœci optymalne ustawienie wartoœci szybkoœci uczenia siê
% mo¿na otrzymaæ podczas procesu treningowego. Otrzymane wartoœci b³êdów s¹
% dla nas zadawalaj¹ce, wynikaj¹ z wprowadzonych zak³óceñ danych ucz¹cych
% i testuj¹ych.
%
% <<lab02_error.PNG>>
% 

%% Mno¿enie
%
% Równanie postaci: 
%
% * 3*x_1 * 2*x_2  = Theta 
%
% W przypadku mno¿enia, analogicznie jak w punkcie 'Dodawanie',
% wygenerowano wektory wejœciowe i wyœciowy ucz¹ce oraz testuj¹ce.
% W tym przypadku jednak (wykonywania operacji mno¿enia) nie ma znaczenia konkretne
% ustawienie wspó³czynników przed zmiennymi x_1 i x_2, poniewaŸ mno¿enie
% jest operacj¹ przemienn¹. Mo¿e uproœciæ równanie do postaci 6*x_1 * x_2  = Theta 

%{
clear all; close all; clc;

sigma = 0.1;    % Odchylenie standardowe
x1_1 = 0;   x1_2 = 10;    % krañce wektora wejœæ 1
x2_1 = 0;   x2_2 = 10;    % krañce wektora wejœæ 2
dx_1 = .1;  dx_2 = .1;    % krok wektorów wejœæ
%}
% Wygenerowanie danych ucz¹cych, X_train oraz T_train

n = 1;
for i=x1_1:dx_1:x1_2                   %iteracja po ca³ym wektorze wejœæ 1
    for j=x2_1:dx_2:x2_2               %iteracja po ca³ym wektorze wejœæ 2
         X_train(1,n) = i + sigma*randn(1,1);
         X_train(2,n) = j + sigma*randn(1,1);
         D_train(1,n) = 3 * X_train(1, n) * 2 * X_train(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

% Przyrost wartoœci wektora testuj¹cego nuron
dx_3 = 0.8; dx_4 = 0.8;

% Wygenerowanie danychtestuj¹cych, X_test oraz T_test
n = 1;
for i = x1_1:dx_3:x1_2                   
    for j = x2_1:dx_4:x2_2               
         X_test(1,n) = i + sigma*randn(1,1);
         X_test(2,n) = j + sigma*randn(1,1);
         D_test(1,n) = 3 * X_test(1, n) * 2 * X_test(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

learningRate = 0.0000001;    % wspó³czynnik szybkoœci uczenia         
net = newlin(X_train, D_train, [0], learningRate);  % konfiguracja sieæ (newlin - neuron liniowy)
net.trainParam.epochs = 1000;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.0000001;
net = init(net); %inicjalizacja sieci
init_W = net.IW{1,1};    
init_W_value = sprintf('Wagi: %10.4f %10.4f ',init_W);
disp('Pocz¹tkowe wartoœci wag po³¹czeñ:');
disp(init_W_value);

% Uczenie sieci danymi ucz¹cymi
net = train(net, X_train, D_train);
weights = net.IW{:,1};
weights_value = sprintf('Wagi: %10.4f %10.4f ',weights);
disp('Wartoœci wag po³¹czeñ sieci nauczonej:');
disp(weights_value);
% view(net); % Schemat zastosowanego neuronu typu liniowego
%%
% 
% <<lab02_perceptron.PNG>>
% 
% Obliczenie b³êdu metod¹ MSE dla danych ucz¹cych
Y_train = net(X_train);
MSE_train = mse(net, D_train, Y_train);
MSE_train_error = sprintf('MSE dla treningu: %10.4f  ',MSE_train);
disp('Wartoœæ b³êdu danych ucz¹cych na wyjœciu neuronu, obliczonego metod¹ najmniejszych kwadratów wynosi:');
disp(MSE_train_error);

Y = net(X_test); % Podanie na wejœcie nuronu danych testuj¹cych
MSE_test = mse(net, D_test, Y); %Mean-squared error
MSE_test_error = sprintf('MSE dla testowania: %10.4f  ',MSE_test);
disp('Wartoœæ b³êdu danych testuj¹ych na wyjœciu neuronu, obliczonego metod¹ najmniejszych kwadratów wynosi:');
disp(MSE_test_error );

% Edycja wymiarów wektorów wejœciowych i wyjœciowego do poprawnego
% wyœwietlenia danych
s = sqrt(length(X_test));
X_test_1 = reshape(X_test(1,:),s,s);
X_test_2 = reshape(X_test(2,:),s,s);
Y_1 = reshape(Y,s,s);

% Wyœwietlenie wyników dzia³ania sieci dla danych testowych
figure(1)
surf(X_test_1,X_test_2, Y_1); % lub mesh
xlabel('Wejœcie 1');
ylabel('Wejœcie 2');
zlabel('Wyjœcie neuronu');
title('Wykres wartoœci na wyjœciu nueronu, gdy na jego wejœcia wprowadzono dane testowe.');
grid on

%Wyznaczenie wektora b³êdu na wyjœciu neuronu
D_1=reshape(D_test,s,s);
error = Y_1-D_1;

figure(2)
surf(error)
title('Wykres b³êdu, jako ró¿nicy wartoœci oczekiwanych  i testowych wektora wyjœciowego.');
xlabel('Wejœcie 1');
ylabel('Wejœcie 2');
zlabel('Amplituda b³êdu na wyjœciu neuronu');
grid on
%%
%
%%
% Podczas wykonanych testów, widoczna jest poprawa minimalizacji wielkoœci
% b³êdu œredniokwadratowego wraz z zwêkszon¹ liczb¹ epok podczas
% testowania. 
% Wartoœci b³êdów wynios³y odpowiednio:
disp(MSE_train_error);
disp(MSE_test_error );

%%
% Wartoœæ b³êdu œredniokwadratowego jest tym mniejsza, im korelacja wartoœci wektorów wejœciowych jest wiêksza
% (tzn. wartoœci bardziej zbli¿one). Gdy wektory wejœciowe by³y takie same, to wartoœci wag  obu wejœæ by³y by sobie równe.
%
% <<lab02_mnozenie.PNG>>
% 
%
% Jednak zgodnie z przypuszczeniami, zastosowana sieæ nie jest w stanie
% rozró¿niæ wektorów wejœciowych w celu wyznaczenia oczekiwanych wag
% po³¹czeñ, które wynios³y odpowiednio: 
disp(weights_value);

%%
% co œwiadczy o tym, ¿e zastosowany neuron typu liniowego nie jest odpowiednim do
% wykonania mno¿enia zmiennych wejœæ



