%% Parameters to tune: rates of snake depredation of eagle (X) and deer (Y)
X = 0.0;
Y = 0.0;
%________________________________________________________________________
%% Not to tune: these are our system dynamics 

% Parameters of the separated predator-prey systems
b1 = 4;
b2 = -1;
b3 = 2;
b4 = -1;
a12 = 0.5;
a21 = -1;
a34 = 0.5;
a43 = -1;

% Parameters that we tune from the snake entering the second system
a23 = -X; % Snake eats deer
a32 = X/2; % Deer is eaten by snake
a24 = -Y; % Snake eats eagle
a42 = Y/2; % Eagle is eaten by snake

% Parameters we don't consider
a11 = 0;
a22 = 0;
a33 = 0;
a44 = 0;
a13 = 0;
a14 = 0;
a41 = 0;
a31 = 0;

%% Matrix definitions

b = [b1; b2; b3; b4];

A = -[a11 a12 a13 a14;
     a21 a22 a23 a24;
     a31 a32 a33 a34;
     a41 a42 a43 a44];


%%
setlmis([]);

C = lmivar(1, [1 0; 1 0; 1 0; 1 0]);

G = newlmi;
lmiterm([G 1 1 C], 1, A, 's'); 

Cl = newlmi;
lmiterm([-Cl 1 1 C], 1, 1); 


lmisys = getlmis;
 
%% 
target = []; options=zeros(1,5);
options(2) = 500;
options(3) = 100;
[tmin, xfeas] = feasp(lmisys, options, target);
Cmat = diag(xfeas(1:4))


