%% Sztuczne sieci neuronowe - laboratorium 2
%
% # Autor: Tomasz K�kol
% # Data: 12.03.2018 - 26.03.2018
% # AGH
% # Kierunek: Informatyka Stosowana
% # Wydzia�: Fizyki i Informatyki Stosowanej
% 
% 

%% Wst�p
% 
% Podczas drugich zaj�� laboratoryjnych zadaniem by�o stworzenie  neuronu typu liniowego, 
% posiadaj�cego umiej�tno�� wykonywania prostych oblicze� dodawania i mno�enia;
% stosuj�c  2 wej�cia neuronu, uwzgl�dniaj�c oczekiwanych warto�ci wag po��cze�.
%

%% Dodawanie
%
% R�wnanie postaci: 
%
% * 3*x_1 + 2*x_2  = Theta 
%
% Oczekiwanym przez nas rezultatem nauczania neuronu jest uzyskanie wag o warto�ci 3 i 2 dla kolejno  wej�cia x_1 oraz x_2.
% W celu przygotowania prostego widoku graficznego,  pocz�tkowe warto�ci wektor�w wej��  x_1 oraz x_2 s� r�wne, zawieraj�c si� w  przedziale (0, 10), z przrostem r�wnym 0.1.
% Nast�pnie, analogicznie jak podczas laboratorium 1, dane ucz�ce zaszumiono z wykorzystaniem znormalizowanej funkcji randn. 
% Warto�� odchylenia standardowego i ampitudy dla zastosowanego szumu to odpowiednio 0.1 i 1.
% Dla przypadku danych testuj�cych zakres warto�ci warto�ci ustawiono taki
% sam, ale zmniejszy�em pr�b� danych testuj�ych, poprzez zwi�kszenie warto�ci przyrostu wektor�w testuj�ych.
% Dla zastosowanego typu sieci, powszechnym wska�nikiem wielko�ci b��du jest b��d �redniokwadratowy(ang. MSE, Mean squared error).
%
% Wyniki przeprowadzonego uczenia sieci:
clear all; close all; clc;

sigma = 0.1;    % Odchylenie standardowe
x1_1 = 0;   x1_2 = 10;    % kra�ce wektora wej�� 1
x2_1 = 0;   x2_2 = 10;    % kra�ce wektora wej�� 2
dx_1 = .1;  dx_2 = .1;    % krok wektor�w wej��

% Wygenerowanie danych ucz�cych, X_train oraz T_train
n = 1;
for i=x1_1:dx_1:x1_2                   %iteracja po ca�ym wektorze wej�� 1
    for j=x2_1:dx_2:x2_2               %iteracja po ca�ym wektorze wej�� 2
         X_train(1,n) = i + sigma*randn(1,1);
         X_train(2,n) = j + sigma*randn(1,1);
         D_train(1,n) = 3 * X_train(1, n) + 2 * X_train(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

% Przyrost warto�ci wektora testuj�cego nuron
dx_3 = 0.8; dx_4 = 0.8;

% Wygenerowanie danychtestuj�cych, X_test oraz T_test
n = 1;
for i = x1_1:dx_3:x1_2                   
    for j = x2_1:dx_4:x2_2               
         X_test(1,n) = i + sigma*randn(1,1);
         X_test(2,n) = j + sigma*randn(1,1);
         D_test(1,n) = 3 * X_test(1, n) + 2 * X_test(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

learningRate = 0.0000001;    % wsp�czynnik szybko�ci uczenia         
net = newlin(X_train, D_train, [0], learningRate);  % konfiguracja sie� (newlin - neuron liniowy)
net.trainParam.epochs = 1000;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.0000001;
net = init(net); %inicjalizacja sieci
init_W = net.IW{1,1};    
init_W_value = sprintf('Wagi: %10.4f %10.4f ',init_W);
disp('Pocz�tkowe warto�ci wag po��cze�:');
disp(init_W_value);

% Uczenie sieci danymi ucz�cymi
net = train(net, X_train, D_train);
weights = net.IW{:,1};
weights_value = sprintf('Wagi: %10.4f %10.4f ',weights);
disp('Warto�ci wag po��cze� sieci nauczonej:');
disp(weights_value);
% view(net); % Schemat zastosowanego neuronu typu liniowego
%%
% 
% <<lab02_perceptron.PNG>>
% 
% Obliczenie b��du metod� MSE dla danych ucz�cych
Y_train = net(X_train);
MSE_train = mse(net, D_train, Y_train);
MSE_train_error = sprintf('MSE: %10.4f  ',MSE_train);
disp('Warto�� b��du danych ucz�cych na wyj�ciu neuronu, obliczonego metod� najmniejszych kwadrat�w wynosi:');
disp(MSE_train_error);

Y = net(X_test); % Podanie na wej�cie nuronu danych testuj�cych
MSE_test = mse(net, D_test, Y); %Mean-squared error
MSE_test_error = sprintf('MSE: %10.4f  ',MSE_test);
disp('Warto�� b��du danych testuj�ych na wyj�ciu neuronu, obliczonego metod� najmniejszych kwadrat�w wynosi:');
disp(MSE_test_error );

% Edycja wymiar�w wektor�w wej�ciowych i wyj�ciowego do poprawnego
% wy�wietlenia danych
s = sqrt(length(X_test));
X_test_1 = reshape(X_test(1,:),s,s);
X_test_2 = reshape(X_test(2,:),s,s);
Y_1 = reshape(Y,s,s);

% Wy�wietlenie wynik�w dzia�ania sieci dla danych testowych
figure(1)
surf(X_test_1,X_test_2, Y_1); % lub mesh
xlabel('Wej�cie 1');
ylabel('Wej�cie 2');
zlabel('Wyj�cie neuronu');
title('Wykres warto�ci na wyj�ciu nueronu, gdy na jego wej�cia wprowadzono dane testowe.');
grid on

%Wyznaczenie wektora b��du na wyj�ciu neuronu
D_1=reshape(D_test,s,s);
error = Y_1-D_1;

figure(2)
surf(error)
title('Wykres b��du, jako r�nicy warto�ci oczekiwanych  i testowych wektora wyj�ciowego.');
xlabel('Wej�cie 1');
ylabel('Wej�cie 2');
zlabel('Amplituda b��du na wyj�ciu neuronu');
grid on

%%
%
%%
% Podczas wykonanych test�w, widoczna jest poprawa minimalizacji wielko�ci
% b��du �redniokwadratowego wraz z zw�kszon� liczb� epok podczas
% testowania. Du�� zmienno�� wynik�w uzyskuje si� poprzez edycj� parametru learning rate -  
% wydajno�� algorytmu jest bardzo wra�liwa na w�a�ciwe ustawienie szybko�ci uczenia si�. 
% Je�li tempo uczenia si� posiada warto�� zbyt du��, algorytm mo�e oscylowa� i sta� si� niestabilny. 
% Je�li szybko�� uczenia si� jest zbyt ma�a, algorytm potrzebuje zbyt wiele czasu (epok), aby prawid�owo pogrupowa� dane podawane na wej�cia.
% Ustalenie optymalnej warto�ci szybko�ci uczenia si� przed treningiem nie jest  w praktyce mo�liwe. 
% W rzeczywisto�ci optymalne ustawienie warto�ci szybko�ci uczenia si�
% mo�na otrzyma� podczas procesu treningowego. Otrzymane warto�ci b��d�w s�
% dla nas zadawalaj�ce, wynikaj� z wprowadzonych zak��ce� danych ucz�cych
% i testuj�ych.
%
% <<lab02_error.PNG>>
% 

%% Mno�enie
%
% R�wnanie postaci: 
%
% * 3*x_1 * 2*x_2  = Theta 
%
% W przypadku mno�enia, analogicznie jak w punkcie 'Dodawanie',
% wygenerowano wektory wej�ciowe i wy�ciowy ucz�ce oraz testuj�ce.
% W tym przypadku jednak (wykonywania operacji mno�enia) nie ma znaczenia konkretne
% ustawienie wsp�czynnik�w przed zmiennymi x_1 i x_2, poniewa� mno�enie
% jest operacj� przemienn�. Mo�e upro�ci� r�wnanie do postaci 6*x_1 * x_2  = Theta 

%{
clear all; close all; clc;

sigma = 0.1;    % Odchylenie standardowe
x1_1 = 0;   x1_2 = 10;    % kra�ce wektora wej�� 1
x2_1 = 0;   x2_2 = 10;    % kra�ce wektora wej�� 2
dx_1 = .1;  dx_2 = .1;    % krok wektor�w wej��
%}
% Wygenerowanie danych ucz�cych, X_train oraz T_train

n = 1;
for i=x1_1:dx_1:x1_2                   %iteracja po ca�ym wektorze wej�� 1
    for j=x2_1:dx_2:x2_2               %iteracja po ca�ym wektorze wej�� 2
         X_train(1,n) = i + sigma*randn(1,1);
         X_train(2,n) = j + sigma*randn(1,1);
         D_train(1,n) = 3 * X_train(1, n) * 2 * X_train(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

% Przyrost warto�ci wektora testuj�cego nuron
dx_3 = 0.8; dx_4 = 0.8;

% Wygenerowanie danychtestuj�cych, X_test oraz T_test
n = 1;
for i = x1_1:dx_3:x1_2                   
    for j = x2_1:dx_4:x2_2               
         X_test(1,n) = i + sigma*randn(1,1);
         X_test(2,n) = j + sigma*randn(1,1);
         D_test(1,n) = 3 * X_test(1, n) * 2 * X_test(2, n) + sigma*randn(1,1);
         n = n + 1;
    end
end

learningRate = 0.0000001;    % wsp�czynnik szybko�ci uczenia         
net = newlin(X_train, D_train, [0], learningRate);  % konfiguracja sie� (newlin - neuron liniowy)
net.trainParam.epochs = 1000;
net.inputWeights{1,1}.initFcn = ('rands');
net.trainParam.goal = 0.0000001;
net = init(net); %inicjalizacja sieci
init_W = net.IW{1,1};    
init_W_value = sprintf('Wagi: %10.4f %10.4f ',init_W);
disp('Pocz�tkowe warto�ci wag po��cze�:');
disp(init_W_value);

% Uczenie sieci danymi ucz�cymi
net = train(net, X_train, D_train);
weights = net.IW{:,1};
weights_value = sprintf('Wagi: %10.4f %10.4f ',weights);
disp('Warto�ci wag po��cze� sieci nauczonej:');
disp(weights_value);
% view(net); % Schemat zastosowanego neuronu typu liniowego
%%
% 
% <<lab02_perceptron.PNG>>
% 
% Obliczenie b��du metod� MSE dla danych ucz�cych
Y_train = net(X_train);
MSE_train = mse(net, D_train, Y_train);
MSE_train_error = sprintf('MSE dla treningu: %10.4f  ',MSE_train);
disp('Warto�� b��du danych ucz�cych na wyj�ciu neuronu, obliczonego metod� najmniejszych kwadrat�w wynosi:');
disp(MSE_train_error);

Y = net(X_test); % Podanie na wej�cie nuronu danych testuj�cych
MSE_test = mse(net, D_test, Y); %Mean-squared error
MSE_test_error = sprintf('MSE dla testowania: %10.4f  ',MSE_test);
disp('Warto�� b��du danych testuj�ych na wyj�ciu neuronu, obliczonego metod� najmniejszych kwadrat�w wynosi:');
disp(MSE_test_error );

% Edycja wymiar�w wektor�w wej�ciowych i wyj�ciowego do poprawnego
% wy�wietlenia danych
s = sqrt(length(X_test));
X_test_1 = reshape(X_test(1,:),s,s);
X_test_2 = reshape(X_test(2,:),s,s);
Y_1 = reshape(Y,s,s);

% Wy�wietlenie wynik�w dzia�ania sieci dla danych testowych
figure(1)
surf(X_test_1,X_test_2, Y_1); % lub mesh
xlabel('Wej�cie 1');
ylabel('Wej�cie 2');
zlabel('Wyj�cie neuronu');
title('Wykres warto�ci na wyj�ciu nueronu, gdy na jego wej�cia wprowadzono dane testowe.');
grid on

%Wyznaczenie wektora b��du na wyj�ciu neuronu
D_1=reshape(D_test,s,s);
error = Y_1-D_1;

figure(2)
surf(error)
title('Wykres b��du, jako r�nicy warto�ci oczekiwanych  i testowych wektora wyj�ciowego.');
xlabel('Wej�cie 1');
ylabel('Wej�cie 2');
zlabel('Amplituda b��du na wyj�ciu neuronu');
grid on
%%
%
%%
% Podczas wykonanych test�w, widoczna jest poprawa minimalizacji wielko�ci
% b��du �redniokwadratowego wraz z zw�kszon� liczb� epok podczas
% testowania. 
% Warto�ci b��d�w wynios�y odpowiednio:
disp(MSE_train_error);
disp(MSE_test_error );

%%
% Warto�� b��du �redniokwadratowego jest tym mniejsza, im korelacja warto�ci wektor�w wej�ciowych jest wi�ksza
% (tzn. warto�ci bardziej zbli�one). Gdy wektory wej�ciowe by�y takie same, to warto�ci wag  obu wej�� by�y by sobie r�wne.
%
% <<lab02_mnozenie.PNG>>
% 
%
% Jednak zgodnie z przypuszczeniami, zastosowana sie� nie jest w stanie
% rozr�ni� wektor�w wej�ciowych w celu wyznaczenia oczekiwanych wag
% po��cze�, kt�re wynios�y odpowiednio: 
disp(weights_value);

%%
% co �wiadczy o tym, �e zastosowany neuron typu liniowego nie jest odpowiednim do
% wykonania mno�enia zmiennych wej��



