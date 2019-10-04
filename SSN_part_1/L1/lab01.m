
%% Sztuczne sieci neuronowe - laboratorium 1
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
% Poni¿sza praca jest 1 czêœci¹ sprawozdania dotycz¹cej pracy wykonanej podczas trzech zajêæ laboratoryjnych.
% Praca jest podzielona na trzy g³ówne rozdzia³y. 
%
%  Neuronu McCullocha-Pittsa charakteryzuj¹:

%%
% 
% * wejœcia - wiele wejœæ X 
% * wyjœcie -pojedyñcze wyjœcie Y
% * wagi - waga ka¿dego wejœcia 
% * suma wa¿ona - obliczana jako suma iloczynów wartoœci podawanych na
% wejœcia i ich wag
% * funkcja aktywacji - jest to funkcji, wed³ug której obliczana jest wartoœæ wyjœcia neuronu (przyk³adowo funkcja progowa)
%
% Wejœcia i wagi s¹ liczbami rzeczywistymi dodatnimi b¹dŸ ujemnymi.
%
% W przypadku, gdy cecha (tj. wejœcie) jest przyczyn¹ pobudzenia neuronu,
% waga takiej cechy bêdzie dodatnia. W sytuacji, gdy cecha posiada charakter hamuj¹cy - waga bêdzie ujemna. Neuron dokonuje sumowania i dopasowania do progu - tj. biasu (Theta). Czêsto
% przyjmuje siê próg Theta jako wagi Wo wraz z wejœciem równym 1.  Interpretacj¹ neuronu na p³aszczyŸnie dwuwymiarowej jest prosta
% rozdzielaj¹ca 2 klasy. Dziêki temu, w przypadku pojedyñczego nueronu, istnieje mo¿liwoœæ podzia³u zbioru na
% 2 podzbiory danch, wzglêdem wyró¿niaj¹cych ich cech.  
% 
% Równanie prostej rozdzielaj¹cej dwie klasy na p³aszczyŸnie 2D: 
%%
%
% * X1*W2 + X2*W2 - Theta = 0
% 
% Równanie prostej rozdzielaj¹cej dwie klasy na p³aszczyŸnie 3D: 
%%
%
% * X1*W2 + X2*W2 +  X3*W3 - Theta = 0
%
%
% Podczas pierwszych zajêæ laboratoryjnych, zajmowaliœmy siê testowaniem
% perceptronu (prostszej sieci neuronowej, sk³adaj¹cej siê z jednego lub wielu niezale¿nych neuronów McCullocha-Pittsa), który umo¿liwia min. stworzenie odwzorowania bramek logicznych typu
% AND, OR oraz XOR, znanych i stosowanych powszechnie min. w elektronice.
% 
% Dla ka¿dego z przypadków funkcji logicznych, jako wejœcia s³u¿¹ 2 wekory wejœcia, postaci:
%%
%
% * Wejscie numer 1 = [0 0 1 1];
% * Wejœcie numer 2 = [0 1 0 1];
% 
% Wyjœciem jest pojedyñczy wektor, zale¿nym od przypadku funkcji logicznej 
%%
%
% * Wyjœcie OR = [0 1 1 1];
% * Wyjœcie AND = [0 0 0 1];
% * Wyjœcie XOR = [0 1 1 0];
%    

%% Funkcja logiczna AND 
clear all; close all; clc;warning off;
X_AND = [0 1 0 1; 0 0 1 1];  % macierz wektroró podawanych na wejœcie perceptronu
D_AND = [0 0 0 1];   % oczekiwane wyniki na wyjœciu perceptronu 
X_AND_2=[0.1 0.9 0.1 0.9; 0.1 0.1 0.9 0.9];  % macierz wektorów danych  poddany testowaniu 
% etap szkolenia nuronu
net = perceptron;   % stworzoeniesieci jako pojedyñczego perceptronu,
net = configure(net,X_AND,D_AND);   % konfiguracja perceptronu
net.IW{1,1};    % IW - input weigths, konfiguracja wagi po³¹czeñ, b - bias
net.b{1};
%view(net)  % podgl¹d wygl¹du schematu perceptronu w nowym oknie
net = train(net,X_AND,D_AND);   % trening perceptronu  danymi ucz¹cymi, zmienia wartoœci  wag po³¹czeñ
Y_AND = net(X_AND);     
% wizualizacja
figure(1)
plotpv(X_AND,Y_AND);    % wyœwietlenie pkt wytrenowanych
plotpc(net.IW{1},net.b{1}); % wyœwietlenie krzywej liniowej dziel¹cej przestrzeæ 2D na 2 podprzestrzenie
hold on
plotpv(X_AND_2, Y_AND);  % próba sprawdzenia nauczonego neuronu na danych tesuj¹cych brak korekcji biasu skutkuje przydzia³em punktu [0.9 0.9] do niechcianej przez nas klasy
title('Wyœwietlenie rezutlatu podzia³u danych testuj¹cych na 2 podzbiory.'); 
figure(2)
net.b{1} = net.b{1} +0.8;                               % korekcja bias'u, 
Y_AND = net(X_AND_2);                         % podanie danych testowanych na wejœcie percetronu                
plotpv(X_AND,Y_AND);                                % wyœwietlenie danych ucz¹cych
hold on
plotpv(X_AND_2, Y_AND);                      % wyœwietlenie danych testuj¹cych
hold on
plotpc(net.IW{1},net.b{1});
title('Korekta biasu zapewnia przydzia³ danej o wspó³rzêdych [0.9 0.9] do oczekiwanej przez nas klasy.')

%%
% Przyk³adowy rezultat treningu neuronu (funkcja logincza AND) - powodzenie
% 100% podzia³u zbioru wejœæ na 2 podzbiory o wartoœciach oczekiwanych 0 lub 1
% <<add2_2.PNG>>
% 
% W celu nauczenia perceptronu regu³y logicznej AND, na wejœcie perceptronu
% wprowadzi³em 50 wartoœci ucz¹cych dla ka¿dej z 4 kombinacji wejœcia (³¹cznie próba 200 wartoœci ucz¹cych).
% Dla wykonania testu, do wartoœci podawanych na wejœcia perceprtonu  dodane zosta³y losowe zak³ócenia,  o odchyleniu standardowym równym 0.10 i amplitudzie 1.
% Dla sprawdzenia skutecznoœci nauczania, wykona³em zdefiniowanie wektora
% wejœciowego testuj¹cego o liczebnoœci 500 wartoœci (zachowuj¹c wartoœci odchylenia standardowego i amplitudy wporawdzanych zak³óceñ),
% dla których nie s¹ znane wartoœci na wyjœciu perceptronu.

clear all; close all; clc;warning off;
% Generacja macierzy danych X_train do nauczania perceprtonu, D - wektor wyjœcia (wartoœci oczekiwanych wzglêdem macierzy wejœcia X_train)
sigma_AND = 0.1;    % odchylenie standardowe - wprowadzenie standaryzowanych zak³óceñ
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
net_AND= train(net_AND,X_train_AND, D_train_AND);    % nauczanie perceptronu danymi ucz¹cymi
Y_AND = net_AND(X_test_AND);    % wyznaczenie wartoœci wyjœcia dla danych testuj¹cych 
IW_AND = net_AND.IW{:,1};   % wspó³czynniki uczenia - wagi
bias_AND = net_AND.b{1};    % wartoœæ biasu
X_AND_1 = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', IW_AND, bias_AND);
disp('Prosta koloru niebieskiego przedstawia prost¹ progow¹ funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych wartoœci wag i biasu.');
disp('Wartoœci wspó³czynników:');
disp(X_AND_1);
%%
% B³¹d dzia³ania perceptronu, wyra¿ony w procentach,  jako iloraz b³êdnych przypisañ danych testowych wzglêdem wszystkich danych testowych. W rzeczywitoœci wartoœci wektora wyjœæ D_test nie jest znane dla danych testuj¹cych !
error_AND = 100*(length(find((Y_AND ~= D_test_AND))) / length(Y_AND));% w [%]

figure(1)
plot3(X_test_AND(1,:),X_test_AND(2,:),Y_AND,'gx');
plotpc(net_AND.IW{1},net_AND.b{1});
%xb=[0, (-1)*bias/weights(1)]; yb=[(-1)*bias/weights(2), 0]; zb=[0 0];
%hold on; plot3(xb,yb,zb)       
title('Wyœwietlenie wyników testowania perceptronu,funkcja logiczna AND, widok 3D ');
grid on

figure(2)  
plotpv(X_test_AND, Y_AND);  %plot3(X_test(1,:),X_test(2,:),Y,'gx');view(2)
plotpc(net_AND.IW{1},net_AND.b{1}); %hold on; plot(xb,yb) % funkcja aktywacji neuronu
title('Wyœwietlenie wyników testowania perceptronu,funkcja logiczna AND, widok 2D ');
grid on

%%
%
% W przypadku, w którym suma iloczynów wartoœci wejœæ perceptronu i wag po³¹czeñ jest wiêksza od wartoœæ progowej ( tj. bias*(-1),
% wyjœcie neuronu ma wartoœæ 1. Dla przedstawionego przypadku, wielkoœæ b³êdu wyznaczenia wartoœci
% wyjœciowych (dla zadanych wartoœci testowych) przez perceptron wynios³a: 
error_AND_disp = sprintf('Wartoœæ b³êdnego podzia³u wartoœci testuj¹cych wynosi: %10.4f  %', error_AND);
disp('Perceptron zadawalaj¹co obliczy³ wartoœæ wyjœcia sieci.');
disp(error_AND_disp);

%% Funkcja logiczna OR
% Do nauczenia perceptronu regu³y logicznej OR, tak jak podano we wstêpie, zastosowano odpowiednie nauczanie z oczekiwanym wekorem wyjœcia D = [ 0 1 1 1]. Dane wejœciowe zosta³y przygotowane analogicznie jak do obs³ugi  regu³y AND. 
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
net_OR = train(net_OR, X_train_OR, D_train_OR);    % nauczanie perceptronu danymi ucz¹cymi
Y_OR = net_OR(X_test_OR);  % wyznaczenie wartoœci wyjœcia dla danych testuj¹cych 
weights_OR = net_OR.IW{:,1};   % wspó³czynniki uczenia - wagi
bias_OR = net_OR.b{1}; % wartoœæ biasu
disp('Prosta koloru niebieskiego przedstawia prost¹ progow¹ funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych wartoœci wag i biasu.');
disp('Wartoœci wspó³czynników:');
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
title('Wyœwietlenie wyników testowania perceptronu, funkcja logiczna OR, widok 3D ');
grid on

figure(2)
plotpv(X_test_OR, Y_OR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpc(net_OR.IW{1},net_OR.b{1});     %hold on; plot(xb,yb, zb); view(2);% funkcja aktywacji neuronu
title('Wyœwietlenie wyników testowania perceptronu, funkcja logiczna OR, widok 2D ');
grid on

%%
% Analogicznie jak we wczeœniejszym przypadku (AND), dla przypadku, w którym suma iloczynów wartoœci wejœæ perceptronu i wag po³¹czeñ jest wiêksza od wartoœæ progowej ( tj. bias*(-1),
% wyjœcie neuronu ma wartoœæ 1. Dla przedstawionego przypadku, wielkoœæ b³êdu wyznaczenia wartoœci
% wyjœciowych (dla zadanych wartoœci testowych) przez perceptron wynios³a: 

error_OR_disp = sprintf('Wartoœæ b³êdnego podzia³u wartoœci testuj¹cych wynosi: %10.4f  [%]', error_OR);
disp('Perceptron zadawalaj¹co obliczy³ wartoœæ wyjœcia sieci.');
disp(error_OR_disp);

%% Funkcja logiczna XOR
% W przypadku rozwi¹zania bramki logicznej XOR, zastosowanie pojedyñczego
% perceptronu nie skutkuje rozdzieleniem 2 klas w zbiorze ze skutecznoœci¹
% 100 %, a w zgrubnym przybli¿eniu oko³o 75%. Jest to spowodowane tym, ¿e dla
% struktury zbudowanej z 1 peerceptronu mo¿liwy jest podzia³ p³aszczyzny
% (widok 2D) jedn¹ prost¹. Opis rozszerza poni¿sza próba wytrenowania
% perceptronu rozwi¹zania funkcji XOR.


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
net_XOR = train(net_XOR, X_train_XOR, D_train_XOR);    % nauczanie perceptronu danymi ucz¹cymi
Y_XOR = net_XOR(X_train_XOR); %X_test_XOR % wyznaczenie wartoœci wyjœcia dla danych testuj¹cych 
weights_XOR = net_XOR.IW{:,1};   % wspó³czynniki uczenia - wagi
bias_XOR = net_XOR.b{1} +0.5; % wartoœæ biasu
disp('Prosta koloru niebieskiego przedstawia prost¹ progow¹ funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych wartoœci wag i biasu.');
disp('Wartoœci wspó³czynników:');
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
title('Wyœwietlenie wyników testowania perceptronu, funkcja logiczna XOR, widok 3D ');
grid on

figure(2)
%plotpv(X_test_XOR, Y_XOR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpv(X_train_XOR, Y_XOR);    %plot3(X_ORtest(1,:),X_ORtest(2,:),Y_OR,'gx');view(2);
plotpc(net_XOR.IW{1},net_XOR.b{1});     %hold on; plot(xb,yb, zb); view(2);% funkcja aktywacji neuronu
title('Wyœwietlenie wyników testowania perceptronu, funkcja logiczna XOR, widok 2D ');
grid on

%%
% Analogicznie jak we wczeœniejszym przypadku (AND), dla przypadku, w którym suma iloczynów wartoœci wejœæ perceptronu i wag po³¹czeñ jest wiêksza od wartoœæ progowej ( tj. bias*(-1),
% wyjœcie neuronu ma wartoœæ 1. Dla przedstawionego przypadku, wielkoœæ b³êdu wyznaczenia wartoœci
% wyjœciowych (dla zadanych wartoœci testowych) przez perceptron wynios³a: 

error_XOR_disp = sprintf('Wartoœæ b³êdnego podzia³u wartoœci testuj¹cych wynosi: %10.4f  %', error_XOR);
disp('Perceptron zadawalaj¹co obliczy³ wartoœæ wyjœcia sieci.');
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
disp('Prosta koloru niebieskiego przedstawia prost¹ progow¹ funkcji aktywacji neuronu.  Jest ona wyliczona na podstawie uzyskanych wartoœci wag i biasu.');
disp('Wartoœci wspó³czynników:');
X_XOR = sprintf('Wag: %10.4f %10.4f , Biasu: %10.2f ', weights, bias);
disp(X_XOR);                                      
plot3(X_test(1,:),X_test(2,:),y,'gx','linewidth',3)
title('Testowania nuronu (gaussowska f.aktywacji), funkcja logiczna XOR, widok 3D ');
xlabel('Wejœcie 1')
ylabel('Wejœcie 2');
zlabel('Wyjœcie');
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
error_disp = sprintf('Wartoœæ b³êdnego podzia³u wartoœci testuj¹cych wynosi: %10.4f  %', error);
disp(error_disp);
% wyœwietlenie wyników dzia³ania neuronu po zastosowaniu progu
figure
plot3(X_test(1,:),X_test(2,:),y,'gx','linewidth',3)
title('Testowania nuronu, funkcja logiczna XOR, widok 3D ');
xlabel('Wejœcie 1')
ylabel('Wejœcie 2');
zlabel('Wyjœcie');
grid on

