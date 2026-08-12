% This script creates a desired airfoil.

% Terminologies to know:
%     - Camber Line: Middle line between the upper and lower surfaces.
%     - Leading Edge: Most forward point on the camber line.
%     - Trailing Edge: Most backward point on the camber line.
%     - Chord Line: Straight line connecting the leading/trailing edge.
%     - Chord Length: Chord line distance.
%     - Local Camber: Any point along the camber line; distance between 
%       the chord line and camber line.
%     - Maximum Camber: Maximum local camber distance from chord line.
%     - Local Thickness: Thickness between the upper and lower surfaces 
%       at any point along the chord line.
%     - Maximum Thickness: Maximum distance between surfaces.;
%     - 4-digit NACA #: First digit defines the maximum camber in percent 
%       of chord, the second digit defines the distance from the leading 
%       edge to the point of maximum camber in tenths of the chord, and 
%       the last two digits defines the maximum thickness in percent of 
%       chord.

clear, clc, close all % Clears variables, command prompt, and images

% Define the chord line since the NACA numbers are based on this line
npoints         = 10001;
chord_length    = 2;
chord_line      = linspace(0,chord_length,npoints);
chord_line_norm = linspace(0,1,npoints);

% Define the 4-digit NACA
NACA = input("Enter the desired 4-digit NACA airfoil: ", 's');

% Check for quitting condition
if ismember(lower(string(NACA)), ["q", "quit"])
    disp("Quitting program... Goodbye!")
    return
end

% Define the max camber from first NACA digit
maxCamber_y = (str2double(NACA(1))/100) * chord_length;
maxCamber_ynorm = maxCamber_y / chord_length; % Normalized

% Define the x-value of where the max camber is located from the LE
maxCamber_x = (str2double(NACA(2))/10) * chord_length;
maxCamber_xnorm = maxCamber_x / chord_length; % Normalized

% Define the max thickness of the airfoil from the last two NACA values
maxThickness = (str2double(NACA(3:4))/100) * chord_length;

% Determine if the user wants the airfoil to be dimensionless or not
airfoil_type = input("Want the airfoil to be dimensionless?: ", "s");

% Check for quitting condition
if ismember(lower(string(airfoil_type)), ["q", "quit"])
    disp("Quitting program... Goodbye!")
    return
end

% Determine if the user wants an open or closed TE
te = input("Do you want an open or closed trailing edge? (open/closed): "...
    , "s");

% Calculate for the airfoil parameters (dimensional/nondimensional)
if ismember(lower(string(airfoil_type)), ["y", "yes"])
    % Find nondimensional camber line and thickness
    if maxCamber_ynorm == 0
        % Preallocate dimensionless camber, thickness, and surface arrays
        camber_line_norm = zeros(size(chord_line_norm));
        thickness_norm = zeros(size(chord_line_norm));
        upper_surface = zeros(size(chord_line_norm));
        lower_surface = zeros(size(chord_line_norm));
        if ismember(lower(string(te)), "open")
            for i = 1:numel(chord_line_norm)
                % Calculating for thickness
                x_norm = chord_line_norm(i);
                thickness_norm(i) = (maxThickness/chord_length) * ...
                    (2.969*sqrt(x_norm) - 1.260*x_norm - 3.516*(x_norm^2) + ...
                    2.843*(x_norm^3) - 1.015*(x_norm^4));
                % Find the upper and lower symmetric surfaces
                upper_surface(i) = 0.5 * thickness_norm(i);
                lower_surface(i) = -0.5 * thickness_norm(i);
            end
            % Plot the related surfaces
            plot(chord_line_norm, upper_surface, LineStyle="-", ...
                Color="white")
            hold on
            plot(chord_line_norm, lower_surface, LineStyle="-", ...
                Color="white")
            plot([chord_line_norm(end), chord_line_norm(end)], ...
                [lower_surface(end), upper_surface(end)], LineStyle="-", ...
                Color="white")
        elseif ismember(lower(string(te)), "closed")
            for i = 1:numel(chord_line_norm)
                % Calculating for thickness
                x_norm = chord_line_norm(i);
                thickness_norm(i) = (maxThickness/chord_length) * ...
                    (2.98*sqrt(x_norm) - 1.32*x_norm - 3.286*(x_norm^2) + ...
                    2.441*(x_norm^3) - 0.815*(x_norm^4));
                % Find the upper and lower symmetric surfaces
                upper_surface(i) = 0.5 * thickness_norm(i);
                lower_surface(i) = -0.5 * thickness_norm(i);
            end
            % Plot the related surfaces
            plot(chord_line_norm, upper_surface, LineStyle="-", ...
                Color="white")
            hold on
            plot(chord_line_norm, lower_surface, LineStyle="-", ...
                Color="white")
        else
            disp("Invalid input. Either type 'open' or 'closed'.")
            return
        end
    else
        % Preallocate dimensionless camber and thickness arrays
        camber_line_norm = zeros(size(chord_line_norm));
        thickness_norm = zeros(size(chord_line_norm));
        upper_surface_x = zeros(size(chord_line_norm));
        upper_surface_y = zeros(size(chord_line_norm));
        lower_surface_x = zeros(size(chord_line_norm));
        lower_surface_y = zeros(size(chord_line_norm));
        dydx = zeros(size(chord_line_norm));
        for i = 1:numel(chord_line_norm)
            x_norm = chord_line_norm(i);
            % Calculating for dimensionless camber line and related slope
            if x_norm <= maxCamber_xnorm
                camber_line_norm(i) = (maxCamber_ynorm) * ...
                    (2*(x_norm/maxCamber_xnorm) - ...
                    (x_norm/maxCamber_xnorm)^2);
                dydx(i) = ((2*maxCamber_ynorm)/maxCamber_xnorm)...
                     * (1 - (x_norm/maxCamber_xnorm));
            else
                camber_line_norm(i) = maxCamber_ynorm * ...
                    (2*((1 - x_norm)/(1 - maxCamber_xnorm)) - ((1 - ...
                    x_norm)/(1 - maxCamber_xnorm))^2);
                dydx(i) = 2*((maxCamber_ynorm)/(1-maxCamber_xnorm))...
                    * (((1 - x_norm)/(1 - maxCamber_xnorm)) - 1);
            end
            % Calculating for dimensionless thickness and surfaces
            if ismember(lower(string(te)), "open")
                thickness_norm(i) = (maxThickness/chord_length) * ...
                    (2.969*sqrt(x_norm) - 1.260*x_norm - 3.516*(x_norm^2) + ...
                    2.843*(x_norm^3) - 1.015*(x_norm^4));
                upper_surface_x(i) = x_norm - (thickness_norm(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                upper_surface_y(i) = camber_line_norm(i) + (...
                    thickness_norm(i)/(2 * sqrt(1+dydx(i)^2)));
                lower_surface_x(i) = x_norm + (thickness_norm(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                lower_surface_y(i) = camber_line_norm(i) - (...
                    thickness_norm(i)/(2 * sqrt(1+dydx(i)^2)));
            elseif ismember(lower(string(te)), "closed")
                thickness_norm(i) = (maxThickness/chord_length) * ...
                    (2.98*sqrt(x_norm) - 1.32*x_norm - 3.286*(x_norm^2) + ...
                    2.441*(x_norm^3) - 0.815*(x_norm^4));
                upper_surface_x(i) = x_norm - (thickness_norm(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                upper_surface_y(i) = camber_line_norm(i) + (...
                    thickness_norm(i)/(2 * sqrt(1+dydx(i)^2)));
                lower_surface_x(i) = x_norm + (thickness_norm(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                lower_surface_y(i) = camber_line_norm(i) - (...
                    thickness_norm(i)/(2 * sqrt(1+dydx(i)^2)));
            else
                disp("Invalid input. Either enter 'open' or 'closed'.")
                return
            end
        end
        % Plot the related surfaces
        plot(upper_surface_x, upper_surface_y, LineStyle="-", ...
            Color="white")
        hold on
        plot(lower_surface_x, lower_surface_y, LineStyle="-", ...
            Color="white")
        if ismember(lower(string(te)), "open")
            plot([lower_surface_x(end), upper_surface_x(end)], ...
                [lower_surface_y(end), upper_surface_y(end)], LineStyle="-", ...
                Color="white")
        end
    end

    % Plot the airfoil characteristics
    plot(chord_line_norm, camber_line_norm, LineStyle=":", Color="white")
    plot(maxCamber_xnorm, maxCamber_ynorm, Marker="o", ...
        MarkerEdgeColor="red", MarkerFaceColor="red")
    grid on
    grid minor
    axis equal
    xlim([0,1])
    xlabel("x/c")
    ylabel("y/c")
    title("Dimensionless NACA " + string(NACA) + " Airfoil")
    if ismember(lower(string(te)), "open")
        legend("Upper Surface", "Lower Surface", "TE Surface", "Camber Line", ...
            "Max Camber", Location="best")
    else
        legend("Upper Surface", "Lower Surface", "Camber Line", ...
            "Max Camber", Location="best")
    end
else
    % Find dimensional camber line
    if maxCamber_y == 0
        % Preallocate dimensional camber, thickness, and surface arrays
        camber_line = zeros(size(chord_line));
        thickness = zeros(size(chord_line));
        upper_surface = zeros(size(chord_line));
        lower_surface = zeros(size(chord_line));
        if ismember(lower(string(te)), "open")
            for i = 1:numel(chord_line)
                % Calculating for dimensional thickness
                x = chord_line(i);
                thickness(i) = maxThickness * (2.969*sqrt(x/...
                    chord_length) - 1.260*(x/chord_length) - 3.516*(...
                    (x/chord_length)^2) + 2.843*((x/chord_length)^3)...
                    - 1.015*((x/chord_length)^4));
                % Find the upper and lower symmetric dimensional surfaces
                upper_surface(i) = 0.5 * thickness(i);
                lower_surface(i) = -0.5 * thickness(i);
            end
            % Plot the related dimensional surfaces
            plot(chord_line, upper_surface, LineStyle="-", Color="white")
            hold on
            plot(chord_line, lower_surface, LineStyle="-", Color="white")
            plot([chord_line(end), chord_line(end)], ...
                [lower_surface(end), upper_surface(end)], LineStyle="-", ...
                Color="white")
        elseif ismember(lower(string(te)), "closed")
            for i = 1:numel(chord_line)
                % Calculating for dimensional thickness
                x = chord_line(i);
                thickness(i) = maxThickness * (2.98*sqrt(x/...
                    chord_length) - 1.32*(x/chord_length) - 3.286*(...
                    (x/chord_length)^2) + 2.441*((x/chord_length)^3)...
                    - 0.815*((x/chord_length)^4));
                % Find the upper and lower symmetric dimensional surfaces
                upper_surface(i) = 0.5 * thickness(i);
                lower_surface(i) = -0.5 * thickness(i);
            end
            % Plot the related surfaces
            plot(chord_line, upper_surface, LineStyle="-", Color="white")
            hold on
            plot(chord_line, lower_surface, LineStyle="-", Color="white")
        else
            disp("Invalid input. Either type 'open' or 'closed'.")
            return
        end
    else
        % Preallocate dimensional camber and thickness arrays
        camber_line = zeros(size(chord_line));
        thickness = zeros(size(chord_line));
        upper_surface_x = zeros(size(chord_line));
        upper_surface_y = zeros(size(chord_line));
        lower_surface_x = zeros(size(chord_line));
        lower_surface_y = zeros(size(chord_line));
        dydx = zeros(size(chord_line));
        for i = 1:numel(chord_line)
            x = chord_line(i);
            % Calculating for dimensional camber line and related slope
            if x <= maxCamber_x
                camber_line(i) = maxCamber_y * ...
                    (2*(x/maxCamber_x) - (x/maxCamber_x)^2);
                dydx(i) = (2 * maxCamber_y / maxCamber_x) * (1 - ...
                    (x/maxCamber_x));
            else
                camber_line(i) = maxCamber_y * ...
                    (2*((chord_length-x)/(chord_length-maxCamber_x)) - ...
                    ((chord_length-x)/(chord_length-maxCamber_x))^2);
                dydx(i) = (2 * maxCamber_y / (chord_length-maxCamber_x))...
                    * (((chord_length-x)/(chord_length-maxCamber_x)) - 1);
            end
            % Calculating for dimensional thickness and surfaces
            if ismember(lower(string(te)), "open")
                thickness(i) = maxThickness * ...
                    (2.969*sqrt(x/chord_length) - 1.260*(x/chord_length)...
                    - 3.516*((x/chord_length)^2) + 2.843*((x/...
                    chord_length)^3) - 1.015*((x/chord_length)^4));
                upper_surface_x(i) = x - (thickness(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                upper_surface_y(i) = camber_line(i) + (...
                    thickness(i)/(2 * sqrt(1+dydx(i)^2)));
                lower_surface_x(i) = x + (thickness(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                lower_surface_y(i) = camber_line(i) - (...
                    thickness(i)/(2 * sqrt(1+dydx(i)^2)));
            elseif ismember(lower(string(te)), "closed")
                thickness(i) = maxThickness * ...
                    (2.98*sqrt(x/chord_length) - 1.32*(x/chord_length)...
                    - 3.286*((x/chord_length)^2) + ...
                    2.441*((x/chord_length)^3) - 0.815*((x/chord_length)^4));
                upper_surface_x(i) = x - (thickness(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                upper_surface_y(i) = camber_line(i) + (...
                    thickness(i)/(2 * sqrt(1+dydx(i)^2)));
                lower_surface_x(i) = x + (thickness(i)/(2*...
                    sqrt(1+dydx(i)^2))) * dydx(i);
                lower_surface_y(i) = camber_line(i) - (...
                    thickness(i)/(2 * sqrt(1+dydx(i)^2)));
            else
                disp("Invalid input. Either enter 'open' or 'closed'.")
                return
            end
        end
        % Plot the related surfaces
        plot(upper_surface_x, upper_surface_y, LineStyle="-", ...
            Color="white")
        hold on
        plot(lower_surface_x, lower_surface_y, LineStyle="-", ...
            Color="white")
        if ismember(lower(string(te)), "open")
            plot([lower_surface_x(end), upper_surface_x(end)], ...
                [lower_surface_y(end), upper_surface_y(end)], LineStyle="-", ...
                Color="white")
        end
    end
    % Plot the airfoil characteristics
    plot(chord_line, camber_line, LineStyle=":", Color="white")
    plot(maxCamber_x, maxCamber_y, Marker="o", MarkerEdgeColor="red", ...
        MarkerFaceColor="red")
    grid on
    grid minor
    axis equal
    xlim([0, chord_length])
    xlabel("x")
    ylabel("y")
    title("NACA " + string(NACA) + " Airfoil")
    if ismember(lower(string(te)), "open")
        legend("Upper Surface", "Lower Surface", "TE Surface", "Camber Line", ...
            "Max Camber", Location="best")
    else
        legend("Upper Surface", "Lower Surface", "Camber Line", ...
            "Max Camber", Location="best")
    end
end