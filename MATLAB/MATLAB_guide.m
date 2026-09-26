% MATLAB MASTER GUIDE
% Practical reference for numerical engineering, analysis, plotting, simulation, and programming.
% 
% This is a core-language guide, not a literal catalog of every function in every installed toolbox. To find complete current documentation, use:
%   help functionName
%   doc functionName
%   lookfor keyword
%   which functionName
%   ver
% 
% ============================================================
% 1. CORE IDEAS
% ============================================================
% - MATLAB is array-first: scalars, vectors, and matrices are all arrays.
% - MATLAB indexing starts at 1.
% - Numeric data is double by default.
% - End a statement with ; to suppress Command Window output.
% - () indexes arrays, {} gets the contents of a cell, and . accesses a structure field or object property.
% - Use .*, ./, and .^ for element-wise operations.
% - Use *, /, \, and ^ for matrix / linear-algebra operations.
% 
% x = 3;                     % Scalar
% v = [1 2 3];               % Row vector
% v = [1; 2; 3];             % Column vector
% A = [1 2; 3 4];            % Matrix
% 
% ============================================================
% 2. ENVIRONMENT AND FILES
% ============================================================
% clc                         % Clear Command Window text
% clear                       % Clear workspace
% clear x                     % Clear variable x
% clearvars -except A B       % Clear all except A and B
% close all                   % Close all figures
% format shortG               % Change display format only
% who                         % List workspace variable names
% whos                        % Names, sizes, classes, memory
% pwd                         % Current folder
% cd folderName               % Change current folder
% ls                          % List files
% dir                         % Detailed file listing
% addpath('folderName')       % Add a directory to MATLAB path
% rmpath('folderName')        % Remove directory from MATLAB path
% save data.mat A B           % Save variables in MAT-file
% load data.mat               % Load variables from MAT-file
% 
% File types:
% - Script (.m): commands that share the base workspace.
% - Function (.m): inputs/outputs with a separate local workspace.
% - Class (.m): one classdef per class file; the file must match the class name.
% - Live script (.mlx): rich text, equations, code, and output.
% 
% ============================================================
% 3. HELP, DEBUGGING, AND INTERRUPTS
% ============================================================
% help plot                   % Short Command Window documentation
% doc plot                    % Open full documentation
% lookfor Fourier             % Search documentation text
% which plot                  % Path to the called function
% which -all mean             % Every matching function on the path
% exist('name','file')        % Test for a file/function
% methods ClassName           % List class methods
% properties object           % List object properties
% 
%  dbstop if error            % Break into debugger when an error occurs
% dbstop in myFile at 25      % Set breakpoint
% dbclear all                 % Remove breakpoints
% dbstack                     % Show function call stack
% dbstep                      % Execute next line in debugger
% dbcont                      % Continue debugger execution
% dbquit                      % Exit debugger
% keyboard                    % Pause code and enter debug prompt
% 
% Interrupt current execution on macOS: Command + .
% Ctrl+C also works in many contexts.
% 
% ============================================================
% 4. BUILDING ARRAYS
% ============================================================
% zeros(3,4)                  % Zero matrix
% ones(3,4)                   % One matrix
% eye(3)                      % Identity matrix
% nan(2,3)                    % NaN array
% inf(2,3)                    % Inf array
% true(2,3)                   % Logical true array
% false(2,3)                  % Logical false array
% rand(3,1)                   % Uniform random numbers [0,1)
% randn(3,1)                  % Standard normal random numbers
% randi(10,3,1)               % Integers from 1 through 10
% 
% 1:5                         % [1 2 3 4 5]
% 0:0.1:1                     % Fixed step range
% linspace(0,1,11)            % 11 points including endpoints
% logspace(0,3,4)             % [1 10 100 1000]
% repmat(A,2,3)               % Tile array A
% reshape(A,m,n)              % Change shape, column-major ordering
% squeeze(A)                  % Remove singleton dimensions
% permute(A,[2 1 3])          % Reorder dimensions
% cat(1,A,B)                  % Concatenate along dimension 1
% [A B]                       % Horizontal concatenation
% [A; B]                      % Vertical concatenation
% 
% Preallocate arrays in substantial loops:
% N = 10000;
% y = zeros(N,1);
% for k = 1:N
%     y(k) = sin(k/10);
% end
% 
% ============================================================
% 5. SIZE AND INDEXING
% ============================================================
% size(A)                     % Array dimensions
% [m,n] = size(A)             % Rows and columns (2-D)
% numel(A)                    % Total element count
% length(v)                   % Largest dimension
% ndims(A)                    % Number of dimensions
% isempty(A)                  % True if array has no elements
% isvector(A); ismatrix(A); isscalar(A)
% 
% A(2,3)                      % Row 2, column 3
% A(:,3)                      % All rows, column 3
% A(2,:)                      % Row 2, all columns
% A(1:3,2:4)                  % Index ranges
% A(end,:)                    % Last row
% A(1:2:end,:)                % Every other row
% A([1 3 5],:)                % Selected rows
% A(:)                        % Every element as a column vector
% A(mask)                     % Logical indexing
% find(mask)                  % Indices of true elements
% 
% mask = A > 0;
% positiveValues = A(mask);
% A(A < 0) = 0;               % Replace negative entries
% 
% MATLAB is column-major: A(:) stacks columns first.
% 
% ============================================================
% 6. ARITHMETIC AND LINEAR ALGEBRA
% ============================================================
% A + B; A - B                % Addition/subtraction
% A * B                       % Matrix multiplication
% A .* B                      % Element-by-element multiplication
% A / B                       % Matrix right division
% A ./ B                      % Element-by-element division
% A \ B                       % Solve A*X = B
% A .\ B                      % Element-by-element left division
% A ^ n                       % Matrix power
% A .^ n                      % Element-by-element power
% A'                          % Complex-conjugate transpose
% A.'                         % Nonconjugate transpose
% 
% x = A \ b                   % Solve A*x=b; preferable to inv(A)*b
% inv(A)                      % Inverse; usually avoid for solves
% pinv(A)                     % Moore-Penrose pseudoinverse
% rank(A); det(A); trace(A)
% norm(v); norm(A); cond(A)
% diag(A); triu(A); tril(A); rref(A)
% lu(A); qr(A); chol(A); svd(A); eig(A)
% [V,D] = eig(A)
% [U,S,V] = svd(A)
% 
% x = linspace(0,2*pi,1000);
% y = sin(x).^2 .* exp(-0.1*x);  % Element-wise sampled function
% 
% ============================================================
% 7. COMMON MATH FUNCTIONS
% ============================================================
% abs(x); sign(x); sqrt(x); nthroot(x,n)
% exp(x); log(x); log10(x); log2(x)
% sin(x); cos(x); tan(x)
% asin(x); acos(x); atan(x); atan2(y,x)
% sinh(x); cosh(x); tanh(x)
% deg2rad(x); rad2deg(x)
% real(z); imag(z); conj(z); angle(z); complex(a,b)
% round(x); floor(x); ceil(x); fix(x)
% mod(a,b); rem(a,b)
% min(A); max(A); sum(A); prod(A); mean(A); median(A)
% std(A); var(A); rms(A)
% cumsum(A); cumprod(A)
% diff(A); gradient(A)
% trapz(x,y); cumtrapz(x,y)
% interp1(x,y,xq); interp2(...); interpn(...)
% 
% mean(A,1)                   % Mean of each column
% mean(A,2)                   % Mean of each row
% sum(A,'all')                % Sum every element
% 
% ============================================================
% 8. LOGICALS AND COMPARISONS
% ============================================================
% A == B; A ~= B
% A < B; A <= B; A > B; A >= B
% A & B; A | B; ~A            % Element-wise logical operations
% p && q; p || q              % Scalar short-circuit operations
% all(mask); any(mask)
% isnan(A); isinf(A); isfinite(A)
% ismember(x, allowed)
% unique(A); intersect(A,B); union(A,B); setdiff(A,B)
% 
% Text comparison:
% answer = input('Enter q to quit: ','s');
% if ismember(lower(string(answer)), ["q", "quit"])
%     return
% end
% 
% Use strcmp or strcmpi for exact text comparison:
% strcmp(s1,s2)               % Case-sensitive
% strcmpi(s1,s2)              % Case-insensitive
% 
% ============================================================
% 9. CONTROL FLOW
% ============================================================
% if condition
%     statements
% elseif otherCondition
%     statements
% else
%     statements
% end
% 
% switch value
%     case 1
%         statements
%     case {2,3}
%         statements
%     otherwise
%         statements
% end
% 
% for k = 1:N
%     statements
% end
% 
% while condition
%     statements
% end
% 
% break                       % End nearest for/while loop
% continue                    % Skip to next loop iteration
% return                      % Exit current script or function
% 
% ============================================================
% 10. FUNCTIONS
% ============================================================
% function y = f(x)
%     y = x.^2;
% end
% 
% function [lift,drag] = forces(q,S,CL,CD)
%     lift = q*S*CL;
%     drag = q*S*CD;
% end
% 
% function area = airfoilArea(x,thickness)
% % AIRFOILAREA Compute area under thickness distribution.
%     area = trapz(x,thickness);
% end
% 
% f = @(x) x.^2 + 2*x + 1;    % Anonymous function
% root = fzero(f,0);          % Root finding
% nargin; nargout             % Input/output argument count
% 
% ============================================================
% 11. TEXT, CELLS, STRUCTURES, TABLES
% ============================================================
% s = "NACA 2412";            % String scalar (modern preferred)
% c = 'NACA 2412';            % Character vector
% strlength(s); contains(s,"2412"); startsWith(s,"NACA")
% split(s," "); replace(s,"NACA","Airfoil")
% str2double("2412"); num2str(3.14159)
% sprintf("CL = %.3f",CL); fprintf("CL = %.3f\n",CL)
% 
% C = {42,"airfoil"; [1 2 3],true};
% C(1,2)                      % Cel