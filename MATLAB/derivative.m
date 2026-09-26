'This checks for derivatives';

clear,clc,close all

syms x ymc xmc c

y = ymc * (2 * ((c-x)/(c-xmc)) - ((c-x)/(c-xmc))^2);

dydx = diff(y,x)