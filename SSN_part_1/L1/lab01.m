
%% Sztuczne sieci neuronowe - laboratorium 1
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
% Poni�sza praca jest 1 cz�ci� sprawozdania dotycz�cej pracy wykonanej podczas trzech zaj�� laboratoryjnych.
% Praca jest podzielona na trzy g��wne rozdzia�y. 
%
%  Neuronu McCullocha-Pittsa charakteryzuj�:

%%
% 
% * wej�cia - wiele wej�� X 
% * wyj�cie -pojedy�cze wyj�cie Y
% * wagi - waga ka�dego wej�cia 
% * suma wa�ona - obliczana jako suma iloczyn�w warto�ci podawanych na
% wej�cia i ich wag
% * funkcja aktywacji - jest to funkcji, wed�ug kt�rej obliczana jest warto�� wyj�cia neuronu (przyk�adowo funkcja progowa)
%
% Wej�cia i wagi s� liczbami rzeczywistymi dodatnimi b�d� ujemnymi.
%
% W przypadku, gdy cecha (tj. wej�cie) jest przyczyn� pobudzenia neuronu,
% waga takiej cechy b�dzie dodatnia. W sytuacji, gdy cecha posiada charakter hamuj�cy - waga b�dzie ujemna. Neuron dokonuje sumowania i dopasowania do progu - tj. biasu (Theta). Cz�sto
% przyjmuje si� pr�g Theta jako wagi Wo wraz z wej�ciem r�wnym 1.  Interpretacj� neuronu na p�aszczy�nie dwuwymiarowej jest prosta
% rozdzielaj�ca 2 klasy. Dzi�ki temu, w przypadku pojedy�czego nueronu, istnieje mo�liwo�� podzia�u zbioru na
% 2 podzbiory danch, wzgl�dem wyr�niaj�cych ich cech.  
% 
% R�wnanie prostej rozdzielaj�cej dwie klasy na p�aszczy�nie 2D: 
%%
%
% * X1*W2 + X2*W2 - Theta = 0
% 
% R�wnanie prostej rozdzielaj�cej dwie klasy na p�aszczy�nie 3D: 
%%
%
% * X1*W2 + X2*W2 +  X3*W3 - Theta = 0
%
%
% Podczas pierwszych zaj�� laboratoryjnych, zajmowali�my si� testowaniem
% perceptronu (prostszej sieci neuronowej, sk�adaj�cej si� z jednego lub wielu niezale�nych neuron�w McCullocha-Pittsa), kt�ry umo�liwia min. stworzenie odwzorowania bramek logicznych typu
% AND, OR oraz XOR, znanych i stosowanych powszechnie min. w elektronice.
% 
% Dla ka�dego z przypadk�w funkcji logicznych, jako wej�cia s�u�� 2 wekory wej�cia, postaci:
%%
%
% * Wejscie numer 1 = [0 0 1 1];
% * Wej�cie numer 2 = [0 1 0 1];
% 
% Wyj�ciem jest pojedy�czy wektor, zale�nym od przypadku funkcji logicznej 
%%
%
% * Wyj�cie OR = [0 1 1 1];
% * Wyj�cie AND = [0 0 0 1];
% * Wyj�cie XOR = [0 1 1 0];
%    

%% Funkcja logiczna AND 
clear all; close all; clc;warning off;
X_AND = [0 1 0 1; 0 0 1 1];  % macierz wektror� podawanych na wej�cie perceptronu
D_AND = [0 0 0 1];   % oczekiwane wyniki na wyj�ciu perceptronu 
X_AND_2=[0.1 0.9 0.1 0.9; 0.1 0.1 0.9 0.9];  % macierz wektor�w danych  poddany testowaniu 
% etap szkolenia nuronu
net = perceptron;   % stworzoeniesieci jako pojedy�czego perceptronu,
net = configure(net,X_AND,D_AND);   % konfiguracja perceptronu
net.IW{1,1};    % IW - input weigths, konfiguracja wagi po��cze�, b - bias
net.b{1};
%view(net)  % podgl�d wygl�du schematu perceptronu w nowym oknie
net = train(net,X_AND,D_AND);   % trening perceptronu  danymi ucz�cymi, zmienia warto�ci  wag po��cze�
Y_AND = net(X_AND);     
% wizualizacja
figure(1)
plotpv(X_AND,Y_AND);    % wy�wietlenie pkt wytrenowanych
plotpc(net.IW{1},net.b{1}); % wy�wietlenie krzywej liniowej dziel�cej przestrze� 2D na 2 podprzestrzenie
hold on
plotpv(X_AND_2, Y_AND);  % pr�ba sprawdzenia nauczonego neuronu na danych tesuj�cych brak korekcji biasu skutkuje przydzia�em punktu [0.9 0.9] do niechcianej przez nas klasy
title('Wy�wietlenie rezutlatu podzia�u danych testuj�cych na 2 podzbiory.'); 
figure(2)
net.b{1} = net.b{1} +0.8;                               % korekcja bias'u, 
Y_AND = net(X_AND_2);                         % podanie danych testowanych na wej�cie percetronu                
plotpv(X_AND,Y_AND);                                % wy�wietlenie danych ucz�cych
hold on
plotpv(X_AND_2, Y_AND);                      % wy�wietlenie danych testuj�cych
hold on
plotpc(net.IW{1},net.b{1});
title('Korekta biasu zapewnia przydzia� danej o wsp�rz�dych [0.9 0.9] do oczekiwanej przez nas klasy.')

%%
% Przyk�adowy rezultat treningu neuronu (funkcja logincza AND) - powodzenie
% 100% podzia�u zbioru wej�� na 2 podzbiory o warto�ciach oczekiwanych 0 lub 1
% <<add2_2.PNG>>
% 
% W celu nauczenia perceptronu regu�y logicznej AND, na wej�cie perceptronu
% wprowadzi�em 50 warto�ci ucz�cych dla ka�dej z 4 kombinacji wej�cia (��cznie pr�ba 200 warto�ci ucz�cych).
% Dla wykonania testu, do warto�ci podawanych na wej�cia perceprtonu  dodane zosta�y losowe zak��cenia,  o odchyleniu standardowym r�wnym 0.10 i amplitudzie 1.
% Dla sprawdzenia skuteczno�ci nauczania, wykona�em zdefiniowanie wektora
% wej�ciowego testuj�cego o liczebno�ci 500 warto�ci (zachowuj�c warto�ci odchylenia standardowego i amplitudy wporawdzanych zak��ce�),
% dla kt�rych nie s� znane warto�ci na wyj�ciu perceptronu.

clear all; close all; clc;warning off;
% Generacja macierzy danych X_train do nauczania perceprtonu, D - wektor wyj�cia (warto�ci oczekiwanych wzgl�dem macierzy wej�cia X_train)
sigma_AND = 0.1;    % odchylenie standardowe - wprowadzenie standaryzowanych zak��ce�
n = 200; 
for i=1:n
    if (i<=n/4)
        X_train_AND(1,i) = sigma_AND*randn(1,1);
        X_train_AND(2,i) = sigma_AND*randn(1,1);
        D_train_AND(1,i) = 0;
    elseif (i<=n/2)
        X_train_AND(1,i) =       sigma_AND*randn(1,1);
        X_train_AND(2,i) = 1 + sigma_AND*randn(1,1);
        D_train_AND(1,i) = 0;
    elseif (i<=3*n/4)
        X_train_AND(1,i) = 1 + sigma_AND*randn(1,1);
        X_train_AND(2,i) =        sigma_AND*randn(1,1);
        D_train_AND(1,i) = 0;
    else
        X_train_AND(1,i) = 1 + sigma_AND*randn(1,1);
        X_train_AND(2,i) = 1 + sigma_AND*randn(1,1);
        D_train_AND(1,i) = 1;
    end
end
% generacja danych do testowania perceptronu:
m = 500;
for i=1:m
    if (i<=m/4)
        X_test_AND(1,i) = sigma_AND*randn(1,1);
        X_test_AND(2,i) = sigma_AND*randn(1,1);
        D_test_AND(1,i) = 0;
    elseif (i<=m/2)
        X_test_AND(1,i) =  sigma_AND*randn(1,1);
        X_test_AND(2,i) = 1 + sigma_AND*randn(1,1);
        D_test_AND(1,i) = 0;
    elseif (i<=3*m/4)
        X_test_AND(1,i) = 1 + sigma_AND*randn(1,1);
        X_test_AND(2,i) = sigma_AND*randn(1,1);
        D_test_AND(1,i) = 0;
    else
        X_test_AND(1,i) = 1 + sigma_AND*randn(1,1);
        X_test_AND(2,i) = 1 + sigma_AND*randn(1,1);
        D_test_AND(1,i) = 1;
    end
end
net_AND = perceptron;
net_AND= train(net_AND,X_train_AND, D_train_AND);    % nauczanie perceptronu danymi ucz�cymi
Y_AND = net_AND(X_test_AND);    % wyznaczenie warto�ci wyj�cia dla danych testuj�cych 
IW_AND = net_AND.IW{:,1};   % wsp�czynniki uczenia - wagi
bias_AND = net_AND.b{1};    % warto�� biasu
X_AND_1 = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', IW_AND, bias_AND);
disp('Prosta koloru niebieskiego przedstawia prost� progow� funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych warto�ci wag i biasu.');
disp('Warto�ci wsp�czynnik�w:');
disp(X_AND_1);
%%
% B��d dzia�ania perceptronu, wyra�ony w procentach,  jako iloraz b��dnych przypisa� danych testowych wzgl�dem wszystkich danych testowych. W rzeczywito�ci warto�ci wektora wyj�� D_test nie jest znane dla danych testuj�cych !
error_AND = 100*(length(find((Y_AND ~= D_test_AND))) / length(Y_AND));% w [%]

figure(1)
plot3(X_test_AND(1,:),X_test_AND(2,:),Y_AND,'gx');
plotpc(net_AND.IW{1},net_AND.b{1});
%xb=[0, (-1)*bias/weights(1)]; yb=[(-1)*bias/weights(2), 0]; zb=[0 0];
%hold on; plot3(xb,yb,zb)       
title('Wy�wietlenie wynik�w testowania perceptronu,funkcja logiczna AND, widok 3D ');
grid on

figure(2)  
plotpv(X_test_AND, Y_AND);  %plot3(X_test(1,:),X_test(2,:),Y,'gx');view(2)
plotpc(net_AND.IW{1},net_AND.b{1}); %hold on; plot(xb,yb) % funkcja aktywacji neuronu
title('Wy�wietlenie wynik�w testowania perceptronu,funkcja logiczna AND, widok 2D ');
grid on

%%
%
% W przypadku, w kt�rym suma iloczyn�w warto�ci wej�� perceptronu i wag po��cze� jest wi�ksza od warto�� progowej ( tj. bias*(-1),
% wyj�cie neuronu ma warto�� 1. Dla przedstawionego przypadku, wielko�� b��du wyznaczenia warto�ci
% wyj�ciowych (dla zadanych warto�ci testowych) przez perceptron wynios�a: 
error_AND_disp = sprintf('Warto�� b��dnego podzia�u warto�ci testuj�cych wynosi: %10.4f  %', error_AND);
disp('Perceptron zadawalaj�co obliczy� warto�� wyj�cia sieci.');
disp(error_AND_disp);

%% Funkcja logiczna OR
% Do nauczenia perceptronu regu�y logicznej OR, tak jak podano we wst�pie, zastosowano odpowiednie nauczanie z oczekiwanym wekorem wyj�cia D = [ 0 1 1 1]. Dane wej�ciowe zosta�y przygotowane analogicznie jak do obs�ugi  regu�y AND. 
%clear all; close all; clc;warning off;
sigma_OR = 0.1;  
% generacja danych do uczenia perceptronu:
n_OR = 100;      
for i=1:n_OR                                          
    if (i<=n_OR/4)
        X_train_OR(1,i) = sigma_OR*randn(1,1);
        X_train_OR(2,i) = sigma_OR*randn(1,1);
        D_train_OR(1,i) = 0;
    elseif (i<=n_OR/2)
        X_train_OR(1,i) =       sigma_OR*randn(1,1);
        X_train_OR(2,i) = 1 + sigma_OR*randn(1,1);
        D_train_OR(1,i) = 1;
    elseif (i<=3*n_OR/4)
        X_train_OR(1,i) = 1 + sigma_OR*randn(1,1);
        X_train_OR(2,i) =        sigma_OR*randn(1,1);
        D_train_OR(1,i) = 1;
    else
        X_train_OR(1,i) = 1 + sigma_OR*randn(1,1);
        X_train_OR(2,i) = 1 + sigma_OR*randn(1,1);
        D_train_OR(1,i) = 1;
    end
end
% generacja danych do testowania perceptronu:
m_OR = 1000;                                                           
for i=1:m_OR
    if (i<=m_OR/4)
        X_test_OR(1,i) = 0 + sigma_OR*randn(1,1);
         X_test_OR(2,i) = 0 + sigma_OR*randn(1,1);
        D_test_OR(1,i) = 0;
    elseif (i<=m_OR/2)
         X_test_OR(1,i) = 0 + sigma_OR*randn(1,1);
         X_test_OR(2,i) = 1 + sigma_OR*randn(1,1);
        D_test_OR(1,i) = 1;
    elseif (i<=3*m_OR/4)
        X_test_OR(1,i) = 1 + sigma_OR*randn(1,1);
         X_test_OR(2,i) = 0 + sigma_OR*randn(1,1);
        D_test_OR(1,i) = 1;
    else
         X_test_OR(1,i) = 1 + sigma_OR*randn(1,1);
         X_test_OR(2,i) = 1 + sigma_OR*randn(1,1);
        D_test_OR(1,i) = 1;
    end
end
net_OR = perceptron;
net_OR = train(net_OR, X_train_OR, D_train_OR);    % nauczanie perceptronu danymi ucz�cymi
Y_OR = net_OR(X_test_OR);  % wyznaczenie warto�ci wyj�cia dla danych testuj�cych 
weights_OR = net_OR.IW{:,1};   % wsp�czynniki uczenia - wagi
bias_OR = net_OR.b{1}; % warto�� biasu
disp('Prosta koloru niebieskiego przedstawia prost� progow� funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych warto�ci wag i biasu.');
disp('Warto�ci wsp�czynnik�w:');
X_OR_1 = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', weights_OR, bias_OR);
disp(X_OR_1);                                      
error_OR = 100*(length(find((Y_OR ~= D_test_OR))) / length(Y_OR));   % w [%]
figure(1)
plot3(X_test_OR(1,:),X_test_OR(2,:),Y_OR,'gx');
plotpc(net_OR.IW{1},net_OR.b{1});
%xb=[bias_OR/weights_OR(1), 0, (-1)*bias_OR/weights_OR(1), (-2)*bias_OR/weights_OR(1)];
%yb=[(-2)*bias_OR/weights_OR(2),(-1)*bias_OR/weights_OR(2), 0, bias_OR/weights_OR(2)];
%zb=[0 0 0 0];
%hold on; plot3(xb,yb,zb)       
title('Wy�wietlenie wynik�w testowania perceptronu, funkcja logiczna OR, widok 3D ');
grid on

figure(2)
plotpv(X_test_OR, Y_OR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpc(net_OR.IW{1},net_OR.b{1});     %hold on; plot(xb,yb, zb); view(2);% funkcja aktywacji neuronu
title('Wy�wietlenie wynik�w testowania perceptronu, funkcja logiczna OR, widok 2D ');
grid on

%%
% Analogicznie jak we wcze�niejszym przypadku (AND), dla przypadku, w kt�rym suma iloczyn�w warto�ci wej�� perceptronu i wag po��cze� jest wi�ksza od warto�� progowej ( tj. bias*(-1),
% wyj�cie neuronu ma warto�� 1. Dla przedstawionego przypadku, wielko�� b��du wyznaczenia warto�ci
% wyj�ciowych (dla zadanych warto�ci testowych) przez perceptron wynios�a: 

error_OR_disp = sprintf('Warto�� b��dnego podzia�u warto�ci testuj�cych wynosi: %10.4f  [%]', error_OR);
disp('Perceptron zadawalaj�co obliczy� warto�� wyj�cia sieci.');
disp(error_OR_disp);

%% Funkcja logiczna XOR
% W przypadku rozwi�zania bramki logicznej XOR, zastosowanie pojedy�czego
% perceptronu nie skutkuje rozdzieleniem 2 klas w zbiorze ze skuteczno�ci�
% 100 %, a w zgrubnym przybli�eniu oko�o 75%. Jest to spowodowane tym, �e dla
% struktury zbudowanej z 1 peerceptronu mo�liwy jest podzia� p�aszczyzny
% (widok 2D) jedn� prost�. Opis rozszerza poni�sza pr�ba wytrenowania
% perceptronu rozwi�zania funkcji XOR.


%% Funkcja logicznaXOR
%clear all; close all; clc;;
sigma_XOR = 0.05;  
% generacja danych do uczenia perceptronu:
n_XOR = 100;      
for i=1:n_XOR                                          
    if (i<=n_XOR/4)
        X_train_XOR(1,i) = sigma_XOR*randn(1,1);
       X_train_XOR(2,i) = sigma_XOR*randn(1,1);
        D_train_XOR(1,i) = 0;
    elseif (i<=n_XOR/2)
        X_train_XOR(1,i) =       sigma_XOR*randn(1,1);
        X_train_XOR(2,i) = 1 + sigma_XOR*randn(1,1);
        D_train_XOR(1,i) = 1;
    elseif (i<=3*n_XOR/4)
        X_train_XOR(1,i) = 1 + sigma_XOR*randn(1,1);
        X_train_XOR(2,i) =        sigma_XOR*randn(1,1);
        D_train_XOR(1,i) = 1;
    else
        X_train_XOR(1,i) = 1 + sigma_XOR*randn(1,1);
        X_train_XOR(2,i) = 1 + sigma_XOR*randn(1,1);
        D_train_XOR(1,i) = 0;
    end
end
% generacja danych do testowania perceptronu:
m_XOR = 1000;                                                           
for i=1:m_XOR
    if (i<=m_XOR/4)
        X_test_XOR(1,i) = 0 + sigma_XOR*randn(1,1);
         X_test_XOR(2,i) = 0 + sigma_XOR*randn(1,1);
        D_test_XOR(1,i) = 0;
    elseif (i<=m_XOR/2)
         X_test_XOR(1,i) = 0 + sigma_XOR*randn(1,1);
         X_test_XOR(2,i) = 1 + sigma_XOR*randn(1,1);
        D_test_XOR(1,i) = 1;
    elseif (i<=3*m_XOR/4)
        X_test_XOR(1,i) = 1 + sigma_XOR*randn(1,1);
         X_test_XOR(2,i) = 0 + sigma_XOR*randn(1,1);
        D_test_XOR(1,i) = 1;
    else
         X_test_XOR(1,i) = 1 + sigma_XOR*randn(1,1);
         X_test_XOR(2,i) = 1 + sigma_XOR*randn(1,1);
        D_test_XOR(1,i) = 0;
    end
end
net_XOR = perceptron;
net_XOR = train(net_XOR, X_train_XOR, D_train_XOR);    % nauczanie perceptronu danymi ucz�cymi
Y_XOR = net_XOR(X_train_XOR); %X_test_XOR % wyznaczenie warto�ci wyj�cia dla danych testuj�cych 
weights_XOR = net_XOR.IW{:,1};   % wsp�czynniki uczenia - wagi
bias_XOR = net_XOR.b{1} +0.5; % warto�� biasu
disp('Prosta koloru niebieskiego przedstawia prost� progow� funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych warto�ci wag i biasu.');
disp('Warto�ci wsp�czynnik�w:');
X_XOR_1 = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', weights_XOR, bias_XOR);
disp(X_XOR_1);                                      
error_XOR = 100*(length(find((Y_XOR ~= D_train_XOR))) / length(Y_XOR));   % w [%]%D_test_XOR
figure(1)
%plot3(X_test_XOR(1,:),X_test_XOR(2,:),Y_XOR,'gx');
plot3(X_train_XOR(1,:),X_train_XOR(2,:),Y_XOR,'gx','linewidth',2);
plotpc(net_XOR.IW{1},net_XOR.b{1});

%xb=[bias_OR/weights_OR(1), 0, (-1)*bias_OR/weights_OR(1), (-2)*bias_OR/weights_OR(1)];
%yb=[(-2)*bias_OR/weights_OR(2),(-1)*bias_OR/weights_OR(2), 0, bias_OR/weights_OR(2)];
%zb=[0 0 0 0];
%hold on; plot3(xb,yb,zb)       
title('Wy�wietlenie wynik�w testowania perceptronu, funkcja logiczna XOR, widok 3D ');
grid on

figure(2)
%plotpv(X_test_XOR, Y_XOR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpv(X_train_XOR, Y_XOR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpc(net_XOR.IW{1},net_XOR.b{1});     %hold on; plot(xb,yb, zb); view(2);% funkcja aktywacji neuronu
title('Wy�wietlenie wynik�w testowania perceptronu, funkcja logiczna XOR, widok 2D ');
grid on

%%
% Analogicznie jak we wcze�niejszym przypadku (AND), dla przypadku, w kt�rym suma iloczyn�w warto�ci wej�� perceptronu i wag po��cze� jest wi�ksza od warto�� progowej ( tj. bias*(-1),
% wyj�cie neuronu ma warto�� 1. Dla przedstawionego przypadku, wielko�� b��du wyznaczenia warto�ci
% wyj�ciowych (dla zadanych warto�ci testowych) przez perceptron wynios�a: 

error_XOR_disp = sprintf('Warto�� b��dnego podzia�u warto�ci testuj�cych wynosi: %10.4f  %', error_XOR);
disp('Perceptron zadawalaj�co obliczy� warto�� wyj�cia sieci.');
disp(error_XOR_disp);


%clear all;close all;clc

szum = 0.1;  
daneUczace = 100;
for i=1:daneUczace
    if (i<=daneUczace/4)
        X_uczenie(1,i) = 0 + szum*randn(1,1);
        X_uczenie(2,i) = 0 + szum*randn(1,1);
        D_uczenie(1,i) = 0;
    elseif (i<=daneUczace/2)
        X_uczenie(1,i) = 0 + szum*randn(1,1);
        X_uczenie(2,i) = 1 + szum*randn(1,1);
        D_uczenie(1,i) = 1;
    elseif (i<=3*daneUczace/4)
        X_uczenie(1,i) = 1 + szum*randn(1,1);
        X_uczenie(2,i) = 0 + szum*randn(1,1);
        D_uczenie(1,i) = 1;
    else
        X_uczenie(1,i) = 1 + szum*randn(1,1);
        X_uczenie(2,i) = 1 + szum*randn(1,1);
        D_uczenie(1,i) = 0;
    end
end

daneTestujace = 1000;
for i=1:daneTestujace
    if (i<=daneTestujace/4)
        X_test(1,i) = 0 + szum*randn(1,1);
        X_test(2,i) = 0 + szum*randn(1,1);
        D_test(1,i) = 0;
    elseif (i<=daneTestujace/2)
        X_test(1,i) = 0 + szum*randn(1,1);
        X_test(2,i) = 1 + szum*randn(1,1);
        D_test(1,i) = 1;
    elseif (i<=3*daneTestujace/4)
        X_test(1,i) = 1 + szum*randn(1,1);
        X_test(2,i) = 0 + szum*randn(1,1);
        D_test(1,i) = 1;
    else
        X_test(1,i) = 1 + szum*randn(1,1);
        X_test(2,i) = 1 + szum*randn(1,1);
        D_test(1,i) = 0;
    end
end

net = feedforwardnet(1);%neuron
net.layers{1}.transferFcn = 'radbas';% funkcja aktywacji Gaussa
net = train(net,X_uczenie,D_uczenie);
y = net(X_test);
weights = net.IW{:,1};
bias = net.b{1};
disp('Prosta koloru niebieskiego przedstawia prost� progow� funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych warto�ci wag i biasu.');
disp('Warto�ci wsp�czynnik�w:');
X_XOR = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', weights, bias);
disp(X_XOR);                                      
plot3(X_test(1,:),X_test(2,:),y,'gx','linewidth',3)
title('Testowania nuronu (gaussowska f.aktywacji), funkcja logiczna XOR, widok 3D ');
xlabel('Wej�cie 1')
ylabel('Wej�cie 2');
zlabel('Wyj�cie');
grid on
progAktywacji = 0.5;% funkcja progowa
for i=1:daneTestujace
    if (y(i)>=progAktywacji)
        y(i) = 1;
    else
        y(i) = 0;
    end
end
error = 100*(length(find(y ~= D_test)) / length(y));   % w [%]%D_test_XOR
error_disp = sprintf('Warto�� b��dnego podzia�u warto�ci testuj�cych wynosi: %10.4f  %', error);
disp(error_disp);
% wy�wietlenie wynik�w dzia�ania neuronu po zastosowaniu progu
figure
plot3(X_test(1,:),X_test(2,:),y,'gx','linewidth',3)
title('Testowania nuronu, funkcja logiczna XOR, widok 3D ');
xlabel('Wej�cie 1')
ylabel('Wej�cie 2');
zlabel('Wyj�cie');
grid on

