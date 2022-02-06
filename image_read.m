
x = imread('File_000.png');
x = imresize(x, [12, NaN]);
x = rgb2gray(x);
x = imbinarize(x);

[i,j] = find(x==0);

[a,b] = size(x);
disp(length(i))
imshow(x)

writematrix(i, "x2.txt")
writematrix(j, "y2.txt")

scatter(j, i)

