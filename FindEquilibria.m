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

%% Find the equilibrium
b = [b1; b2; b3; b4];

A = [a11 a12 a13 a14;
     a21 a22 a23 a24;
     a31 a32 a33 a34;
     a41 a42 a43 a44];
 
eq = inv(A) * b

%Loop for studying the effect of changing eigenvalues

n_iter = 10;

eig_tab = cell(1,n_iter);
div_tab = cell(1,n_iter);

figure()

for n = 1:n_iter
    
    X = X + 0.05;
    
    Y = Y + 0.025;
    
    a23 = -X; % Snake eats deer
    a32 = X/2; % Deer is eaten by snake
    a24 = -Y; % Snake eats eagle
    a42 = Y/2; % Eagle is eaten by snake

    
    b = [b1; b2; b3; b4];

    A = [a11 a12 a13 a14;
     a21 a22 a23 a24;
     a31 a32 a33 a34;
     a41 a42 a43 a44];
    
   eq = inv(A) * b;
    
   [check_deval, check_eigs, div] = linearise(eq,A,b);
   
   eig_tab{n} = check_eigs;
   div_tab{n} = div;
   
   %Plot the eigenvalues
   hold on
   string = sprintf('X= %G Y =%G', X, Y);
   plot(eig_tab{n}, 'o','DisplayName', [string]);
   xlabel('Real')
   ylabel('Imaginary')
   legend('show')
  
end

xline(0)
yline(0)

%Check for global stability...to be implemented


%Extension to equilibria investigation 

%Plug in X and Y values and investigate whether equilibria is stable? 

%% Linearise about equlibria. compute eigenvalues and divergence

function[dd_eval, Eigens, div] = linearise(equilibria, A, b)

syms x y z w

dx = x * (b(1) - A(1,1)*x - A(1,2)*y - A(1,3)*z - A(1,4)*w);
dy = y * (b(2) - A(2,1)*x- A(2,2)*y - A(2,3)*z - A(2,4)*w);
dz = z * (b(3) - A(3,1)*x - A(3,2)*y - A(3,3)*z - A(3,4)*w);
dw = w * (b(4) - A(4,1)*x - A(4,2)*y - A(4,3)*z - A(4,4)*w);

%Matrix of partials:

dd = [diff(dx,x) diff(dx,y) diff(dx,z) diff(dx,w);
    diff(dy,x) diff(dy,y) diff(dy,z) diff(dy,w);
    diff(dz,x) diff(dz,y) diff(dz,z) diff(dz,w)
    diff(dw,x) diff(dw,y) diff(dw,z) diff(dw,w);];

%Evaluate the matrix at the equilibrium points: 

dd_eval = ones(4,4);

x0 = equilibria(1);
y0 = equilibria(2);
z0 = equilibria(3);
w0 = equilibria(4);


for i = 1:4
    for j = 1:4
        dd_eval(i,j) = subs(dd(i,j), {x, y, w, z}, {x0, y0, z0, w0});      
    end
end

%Compute the matrix eigenvalues:
Eigens = eigs(dd_eval);

%Compute the divergence
div = diff(dx,x) + diff(dy,y) +  diff(dz,z) + diff(dw,w);
end 


%% Computing the divergence in order to check for a non-existence of the limit cycle

 










