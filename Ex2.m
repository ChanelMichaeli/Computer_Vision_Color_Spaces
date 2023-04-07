%% EX1
clear all
close all
clc
disp('Noam Atias 311394357, Chanel Michaeli 208491787')
%% Q1 - RGB and Grayscale 
%% 1.1
color_image = imread('colours.jpeg');
color_image = double(color_image);
% Z = zeros(size(color_image(:,:,1)));
% O = ones(size(color_image(:,:,1)));
normalize_color_image = (color_image - min(color_image(:)))/(max(color_image(:))-min(color_image(:)));

%% 1.2
figure('Name','1.2')
imshow(normalize_color_image)
title('Normalized colors image')
%% 1.3
R = normalize_color_image(:,:,1);
G = normalize_color_image(:,:,2);
B = normalize_color_image(:,:,3);

figure('Name','RGB Channel')
subplot(1,3,1)
imshow(R);
title('Red Channel')
subplot(1,3,2)
imshow(G);
title('Green Channel')
subplot(1,3,3)
imshow(B);
title('Blue Channel')

%% 1.5 
gray_image = dip_rgb2gray(normalize_color_image);
gray_image_matlab = rgb2gray(normalize_color_image);
figure("Name",'1.5')
subplot(1,2,1)
imshow(gray_image)
title('Gray image converted from RGB by our function')
subplot(1,2,2)
imshow(gray_image)
title('Gray image converted from RGB by MATLAB function')

% Compare by measurement
mse_1 =  immse(gray_image,gray_image_matlab);
ssim_1 = ssim(gray_image,gray_image_matlab);

%% 1.6
manipulated1_color_image = normalize_color_image;
manipulated2_color_image = normalize_color_image;
manipulated3_color_image = normalize_color_image;

% Linear function on Channel 1 (Red)
manipulated1_color_image(:,:,1) = -1 * R + 1; 
manipulated1_color_image(manipulated1_color_image < 0) = 0;
manipulated1_color_image(manipulated1_color_image > 1) = 1;

% Switch Channel 2 and 3 (Green and Blue)
manipulated2_color_image(:,:,2) = B;      
manipulated2_color_image(:,:,3) = G;   

% Gaussian filter 5X5 on B for Channel 3
manipulated3_color_image(:,:,3) = dip_gaussian_filter(B,5,1);  


figure('Name','Manipulated RGB Channel')
subplot(3,2,1)
imshow(manipulated1_color_image(:,:,1));
title('Manipulated Red Channel')
subplot(3,2,2)
imshow(R);
title('Red Channel')

subplot(3,2,3)
imshow(manipulated2_color_image(:,:,2));
title('Manipulated Green Channel')
subplot(3,2,4)
imshow(G);
title('Green Channel')

subplot(3,2,5)
imshow(manipulated3_color_image(:,:,3));
title('Manipulated Blue Channel')
subplot(3,2,6)
imshow(B);
title('Blue Channel')

% Colors
figure('Name','Manipulated RGB Channel (colors)')
subplot(2,2,1)
imshow(normalize_color_image)
title('Original image')

subplot(2,2,2)
imshow(manipulated1_color_image);
title('Manipulated Red Channel')

subplot(2,2,3)
imshow(manipulated2_color_image);
title('Manipulated Green Channel')

subplot(2,2,4)
imshow(manipulated3_color_image);
title('Manipulated Blue Channel')

%% Q2 - Additive vs Subtractive Color space
%% 2.2
Black = min(min(1-R,1-G),1-B);
Cyan = (1-R-Black)./(1-Black);
Magenta= (1-G-Black)./(1-Black);
Yellow = (1-B-Black)./(1-Black);

figure("Name",'CYMK channels')
subplot(2,2,1)
imshow(Cyan)
title('Cayn channel')
subplot(2,2,2)
imshow(Yellow)
title('Yellow channel')
subplot(2,2,3)
imshow(Magenta)
title('Magenta channel')
subplot(2,2,4)
imshow(Black)
title('Black channel')
%% 2.3
displayCYMK(Cyan,Yellow,Magenta,Black)

%% 2.4
figure('Name','Manipulated CYMK Channels')
subplot(2,2,1)
imshowCYMK(Cyan,Yellow,Magenta,Black)
title('Original image - CYMK')

% Linear manipulation
subplot(2,2,2)
C = (-1)*Cyan + 1;
C(C < 0) = 0;
C(C > 1) = 1;
imshowCYMK(C,Yellow,Magenta,Black); 
title('Linear manipulated Cyan Channel')

% Swap channels (Yellow & Cyan) 
subplot(2,2,3)
imshowCYMK(Yellow,Cyan,Magenta,Black); 
title('Manipulated swap Channels')

% Gaussian filter 5X5 on Black Channel  
subplot(2,2,4)
imshowCYMK(Cyan,Yellow,Magenta,dip_gaussian_filter(Black,5,1));
title('Gaussian manipulated black Channel')

%% Q3 - HSV
%% 3.1
% PDF

%% 3.3
rgb_img = imread('colours.jpeg');
[H,S,V] = dip_rgb2hsv(rgb_img);
figure("Name",'HSV');
subplot(2,2,1);
imshow(H);
title('H Channel');
subplot(2,2,2);
imshow(S);
title('S Channel');
subplot(2,2,3);
imshow(V);
title('V Channel');

%% 3.4
% Comparation to MATLAB function

[H1,S1,V1]=rgb2hsv(normalize_color_image);
figure("Name",'MATLAB Function - HSV');
subplot(3,2,1);
imshow(H1);
title('H Channel - MATLAB Function');
subplot(3,2,2);
imshow(H);
title('H Channel');

% figure("Name",'MATLAB Function - HSV');
subplot(3,2,3);
imshow(S1);
title('S Channel - MATLAB Function');
subplot(3,2,4);
imshow(S);
title('S Channel');

% figure("Name",'MATLAB Function - HSV');
subplot(3,2,5);
imshow(V1);
title('V Channel - MATLAB Function');
subplot(3,2,6);
imshow(V);
title('V Channel');

% Compare by measurement
% D_H_trace = zeros(1,360);
D_H = pdist2(H1,H,'euclidean');
D_S = pdist2(S1,S,'euclidean');
D_V = pdist2(V1,V,'euclidean');
D_H(isnan(D_H)) = 0;
D_H_trace = trace(D_H)
D_S_trace = trace(D_S)
D_V_trace = trace(D_V)

%% 3.5
figure('Name','Manipulated HSV Channels')
subplot(2,2,1)
imshowHSV(H,S,V)
title('Original image')

% Linear manipulation
subplot(2,2,2)
imshowHSV(mod(H + 0.5,1),S,V); %180 degrees
title('Linear manipulated Hue Channel - add 180 degrees')

% Linear manipulation
subplot(2,2,3)
v = V*(-0.5)+1;
v(v < 0) = 0;
v(v > 1) = 1;
imshowHSV(H,S,v); 
title('Linear manipulated Value Channel')

% Gaussian filter 5X5 on Value Channel  
subplot(2,2,4)
imshowHSV(H,S,dip_gaussian_filter(V,5,1));
title('Gaussian manipulated Value Channel')

%% 3.6
swap_rgb(:,:,1) = rgb_img(:,:,3);
swap_rgb(:,:,2) = rgb_img(:,:,1);
swap_rgb(:,:,3) = rgb_img(:,:,2);
[H_swap_rgb,S_swap_rgb,V_swap_rgb] = dip_rgb2hsv(swap_rgb);

figure("Name",'3.6');
subplot(2,2,1);
imshow(H_swap_rgb);
title('H swap RGB Channel');
subplot(2,2,2);
imshow(S_swap_rgb);
title('S swap RGB Channel');
subplot(2,2,3);
imshow(V_swap_rgb);
title('V swap RGB Channel');

%% Q4 - L*a*b
%% 4.1
% PDF

%% 4.2
L_a_b_img = rgb2lab(normalize_color_image);
L = L_a_b_img(:,:,1);
a = L_a_b_img(:,:,2);
b = L_a_b_img(:,:,3);

L_norm = (L - min(L(:)))/(max(L(:))-min(L(:)));
a_norm = (a - min(a(:)))/(max(a(:))-min(a(:)));
b_norm = (b - min(b(:)))/(max(b(:))-min(b(:)));

figure("Name",'L*a*b* channels')
subplot(2,2,1)
imshow(L_norm)
title('L*a*b* color space - Channel L')
subplot(2,2,2)
imshow(a_norm)
title('L*a*b* color space - Channel a')
subplot(2,2,3)
imshow(b_norm)
title('L*a*b* color space - Channel b')

%% 4.3
figure('Name','Manipulated L*a*b Channels')
subplot(2,2,1)
imshowLab(L,a,b)
title('Original image')

% Linear manipulation
subplot(2,2,2)
imshowLab(L,a*(-1)+1,b); 
title('Linear manipulated a Channel')

% swap chanels a and b
subplot(2,2,3)
imshowLab(L,b,a)
title('Manipulated swap Channels a and b')

% Gaussian filter 5X5 on b Channel  
subplot(2,2,4)
imshowLab(dip_gaussian_filter(L,5,1),a,b);
title('Gaussian manipulated L Channel')

%% 4.4
%compare Lab to HSV
figure('Name','compare Lab to HSV')                               
subplot(2,3,1); 
imshow(V);      
title('V channel');
subplot(2,3,2); 
imshow(H);
title('H channel');
subplot(2,3,3); 
imshow(S);
title('S channel');
subplot(2,3,4); 
imshow(L_norm); 
title('L channel');
subplot(2,3,5); 
imshow(a_norm);
title('a channel');
subplot(2,3,6); 
imshow(b_norm);
title('b channel');

%% Q5 - Color Segmentation
%cap 1
cap_1 = imread("cap1.png");
cap_1 = double(cap_1);
normalize_cap_1 = (cap_1 - min(cap_1(:)))/(max(cap_1(:))-min(cap_1(:)));
new_image = Binary_mask(normalize_cap_1,0.61,(2/3));
[x1,y1] = find_loc(new_image,50);
cap1_final = insertShape(normalize_cap_1, 'circle', [x1 y1 20], 'Color', {'red'});
figure('Name','Cap 1');
subplot(3,1,1)
imshow(normalize_cap_1)
title('Original cap 1')
subplot(3,1,2)
imshow(new_image)
title('Binary mask cap 1')
subplot(3,1,3)
imshow(cap1_final)
title('circle the cap 1 of the soda')

%cap 2
cap_2 = imread("cap2.png");
cap_2 = double(cap_2);
normalize_cap_2 = (cap_2 - min(cap_2(:)))/(max(cap_2(:))-min(cap_2(:)));
new_image2 = Binary_mask(normalize_cap_2,0.61,(2/3));
[x2,y2] = find_loc(new_image2,50);
cap2_final = insertShape(normalize_cap_2, 'circle', [x2 y2 20], 'Color', {'red'});
figure('Name','Cap 2');
subplot(3,1,1)
imshow(normalize_cap_2)
title('Original cap 2')
subplot(3,1,2)
imshow(new_image2)
title('Binary mask cap 2')
subplot(3,1,3)
imshow(cap2_final)
title('circle the cap 2 of the soda')

%cap 3
cap_3 = imread("cap3.png");
cap_3 = double(cap_3);
normalize_cap_3 = (cap_3 - min(cap_3(:)))/(max(cap_3(:))-min(cap_3(:)));
new_image3 = Binary_mask(normalize_cap_3,0.61,(2/3));
[x3,y3] = find_loc(new_image3,50);
cap3_final = insertShape(normalize_cap_3, 'circle', [x3 y3 25], 'Color', {'red'});
figure('Name','Cap 3');
subplot(3,1,1)
imshow(normalize_cap_3)
title('Original cap 3')
subplot(3,1,2)
imshow(new_image3)
title('Binary mask cap 3')
subplot(3,1,3)
imshow(cap3_final)
title('circle the cap 3 of the soda')

%% Functions
% 1.4
function Gray = dip_rgb2gray(img)
    Gray = 0.2989 * img(:,:,1) + 0.587 * img(:,:,2) + 0.114 * img(:,:,3);
end

% Function from Ex1 - gaussian filter
function blured_img = dip_gaussian_filter(img, k, sigma)
    blured_img = zeros(height(img),width(img)); 
    temp = padarray(img,[floor(k/2) floor(k/2)],0,'both');
    [X, Y] = meshgrid(-floor(k/2):floor(k/2),-floor(k/2):floor(k/2));
    gaussian_fil =(1/(2*pi*(sigma^2)))*exp(-(X.^2+Y.^2)./(2*(sigma^2)));
    gaussian_fil=gaussian_fil./sum(gaussian_fil,'all');
    for i = ceil(k/2):height(img)-floor(k/2)
        for j = ceil(k/2):width(img)-floor(k/2)
            blured_img(i,j)=sum(gaussian_fil.*temp(i-floor(k/2):i+floor(k/2),j-floor(k/2):j+floor(k/2)),'All');
        end
    end
end

% 3.2
function [H,S,V] = dip_rgb2hsv(rgb_img)

rgb_img = double(rgb_img);
normalize_rgb_image = (rgb_img - min(rgb_img(:)))/(max(rgb_img(:))-min(rgb_img(:)));
R = normalize_rgb_image(:,:,1);
G = normalize_rgb_image(:,:,2);
B = normalize_rgb_image(:,:,3);
C_max = max(normalize_rgb_image,[],3);
C_min = min(normalize_rgb_image,[],3);
Delta = C_max - C_min;

H = zeros(size(rgb_img,1),size(rgb_img,2));
H(Delta == 0) = 0;
H(C_max == R) = 1/6*(mod(((G(C_max == R) - B(C_max == R))./Delta(C_max == R)),6));
H(C_max == G) = 1/6*(((B(C_max == G) - R(C_max == G))./Delta(C_max == G)) + 2);
H(C_max == B) = 1/6*(((R(C_max == B) - G(C_max == B))./Delta(C_max == B)) + 4);

S = zeros(size(rgb_img,1),size(rgb_img,2));
S(C_max ~= 0) = Delta(C_max ~= 0)./C_max(C_max ~= 0);

V = C_max;
end

% 5.1
function find_hsv_range = Binary_mask(img,min_range,max_range)
    [H,S,V] = rgb2hsv(img);
    H_range = H;
    H_range(H < min_range) = 0;
    H_range(H > max_range) = 0;
    filter = median_filter(H_range,3);
    filter(filter ~= 0) = 1;
    find_hsv_range = filter;
end


function [x,y] = find_loc(img,k)
    row = size(img,1)/k;
    column = size(img,2)/k;
    filter = ones(row,column);
    c = 0;
    for i = 1:k
        for j = 1:k
            new = sum(filter.*(img(1+row*(i-1):i*row,1+column*(j-1):j*column)),'all');
            if new > c
                y = (row*(i-1))+(row/2);
                x = (column*(j-1))+(column/2);
                c = new;
            end
        end
    end
end

function median = median_filter(img,k) 
    median = zeros(height(img),width(img)); 
    temp = padarray(img,[floor(k/2) floor(k/2)],0,'both');
    img_ones = ones(height(img),width(img));
    padded_img_ones= padarray(img_ones,[floor(k/2) floor(k/2)],0,'both');
    for j = ceil(k/2):height(img)+floor(k/2)
        for i = ceil(k/2):width(img)+floor(k/2)
            filter = temp(j-floor(k/2):j+floor(k/2),i-floor(k/2):i+floor(k/2));
            filter_ones = padded_img_ones(j-floor(k/2):j+floor(k/2),i-floor(k/2):i+floor(k/2));
            f = nonzeros(filter + filter_ones)-1;
            while true
                finished = 1;
                for s=1:height(f)-1
                    if f(s)>f(s+1)
                        c=f(s);
                        f(s)=f(s+1);
                        f(s+1)=c;
                        finished = 0;
                    end
                end
                if finished
                    break
                end
            end
            [X,Y] = ismember(filter(ceil(k/2),ceil(k/2)),f);
            f(Y(X)) = [];
            if rem(height(f), 2) == 1
                median(j-floor(k/2),i-floor(k/2)) = f(ceil(height(f)/2));
            else
                median(j-floor(k/2),i-floor(k/2)) = (f((height(f)/2)+1)+f((height(f)/2)))/2;
            end
        end
     end
end