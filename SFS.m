clear all
clc
%%
%read data and to split
%ref : https://www.mathworks.com/help/matlab/ref/fgetl.html
file = fopen('prostate_preprocessed.txt');
data = [];
%status = feof(fileID) returns the status of the end-of-file indicator. The feof function returns a 1 if a previous operation set the end-of-file indicator for the specified file. Otherwise, feof returns a 0.
%ref: https://www.mathworks.com/help/matlab/ref/feof.html
while ~feof(file)
    tline = fgetl(file);
	line = split(tline, " ");%ref: https://www.mathworks.com/help/matlab/ref/split.html#bugc8gx-1-newStr
	data = [data line];
    
end
% X = str2double(str) converts the text in str to double precision values. str contains text that represents real or complex numeric values. str can be a character vector, a cell array of character vectors, or a string array. If str is a character vector or string scalar, then X is a numeric scalar. If str is a cell array of character vectors or a string array, then X is a numeric array that is the same size as str.
% Text that represents a number can contain digits, a comma (thousands separator), a decimal point, a leading + or - sign, an e preceding a power of 10 scale factor, and an i or a j for a complex unit. You cannot use a period as a thousands separator, or a comma as a decimal point.
% If str2double cannot convert text to a number, then it returns a NaN value.
[x,y] = size(data);
label = data(:,y);
label(1)= [];
data(:,y) = [];
data(1,:) = [];
dataset = str2double(data(:,1:100));%ref: https://www.mathworks.com/help/matlab/ref/str2double.html

fclose(file);

%%
%labelling
% tf = strcmp(s1,s2) compares s1 and s2 and returns 1 (true) if the two are identical and 0 (false) otherwise. Text is considered identical if the size and content of each are the same. The return result tf is of data type logical.
% The input arguments can be any combination of string arrays, character vectors, and cell arrays of character vectors.
[x,y] = size(dataset);
label_data=[];

for k= 1:x
    a=label(k);
    if strcmp(a,'normal')==1%ref: https://www.mathworks.com/help/matlab/ref/strcmp.html#:~:text=tf%20%3D%20strcmp(%20s1%2Cs2%20)%20compares%20s1%20and%20s2,is%20of%20data%20type%20logical%20.
        flag=0;
    else
        flag=1;
    end
   label_data=[label_data;flag];
end
clear a;

%tic works with the toc function to measure elapsed time. 
%The tic function records the current time, and the toc function uses the recorded value to calculate the elapsed time.

tic
output_sfs = Sequence_forward_selection(dataset, label_data)
disp("Time required to calculate Sequence forward selection: ")
toc


%%
%Sequential Forward Selection (SFS)
function SFS = Sequence_forward_selection(dataset, label)
    %tf = ismember(A, S) returns a vector the same length as A, containing logical 1 (true) where the elements of A are in the set S, and logical 0 (false) elsewhere. In set theory terms, k is 1 where A  S. A and S can be cell arrays of strings.
	%M = max(A) returns the maximum elements of an array.
    %If A is a vector, then max(A) returns the maximum of A.
    %If A is a matrix, then max(A) is a row vector containing the maximum value of each column.
    %If A is a multidimensional array, then max(A) operates along the first array dimension whose size does not equal 1, treating the elements as vectors. The size of this dimension becomes 1 while the sizes of all other dimensions remain the same. If A is an empty array whose first dimension has zero length, then max(A) returns an empty array with the same size as A.
    current_features = [] ;
    subset_list = [];
    size_train = size(dataset,2)
	F1_score = [];
    best_f1 = -1; 
	SFS = []  
	Select = 0; 
	
	for Select=0:size_train 
		Current_f1 = zeros(size_train, 1);
		for i= 1:size_train
			if ismember(i, subset_list)%ref: https://www.mathworks.com/help/stats/dataset.ismember.html
				Current_f1(i) = -1;
				continue
            end
            %disp("add...")
            current_subset = current_features;
			current_subset(end+1) = i;
			current_train = dataset(:,current_subset);
			F1_score_temp = compute_knn(current_train, label);
			Current_f1(i) = F1_score_temp;
		end

		[value, loc_row] = max(Current_f1);%ref: https://www.mathworks.com/help/matlab/ref/max.html#d122e803163
		subset_list(end+1) = loc_row;
		F1_score(end+1) = value;%The special end operator is an easy shorthand way to refer to the last element of vector
		%disp("1")
        if value > best_f1
            %disp("2")
			best_f1 = value;
			SFS = subset_list;

        end
	end
	best_f1
    disp('Select')
    disp(Select)
	state = make_charts(Select+1, F1_score);


end

%Wrapper :
%ClassificationKNN is a nearest-neighbor classification model in which you can alter both the distance metric and the number of nearest neighbors. 
%Because a ClassificationKNN classifier stores training data, you can use the model to compute resubstitution predictions. Alternatively, use the model to classify new observations using the predict method.
%ref : https://www.mathworks.com/help/stats/classificationknn.html
%label = predict(Mdl,X) returns a vector of predicted class labels for the predictor data in the table or matrix X, based on the trained, full or compact classification tree Mdl.
function output_compute = compute_knn(train, label)
	output = [];
    %Cross validation (train: 70%, test: 30%)
	cv = cvpartition(size(train,1),'HoldOut',0.3);
	index = cv.test;
	test_set = train(index);
	test_label = label(index);
	train_set = train(~index);
	train_label = label(~index);			
	classification_model  = fitcknn(train_set,train_label); %create model
	CVMdl = crossval(classification_model,'KFold',5);%cross validation  
	
	for idx=test_set
		prediction = predict(classification_model,idx);%ref: https://www.mathworks.com/help/stats/compactclassificationtree.predict.html#d122e620924
        output = [output prediction];
	
    end
	
	output_compute = compute_performance_measure(test_label, output);
end

%%
function compute_measure = compute_performance_measure(real, predict)
%ref : https://www.mathworks.com/help/matlab/ref/sum.html#d122e1264065
%S = sum(A) returns the sum of the elements of A along the first array dimension whose size does not equal 1.
%If A is a vector, then sum(A) returns the sum of the elements.
%If A is a matrix, then sum(A) returns a row vector containing the sum of each column.
%If A is a multidimensional array, then sum(A) operates along the first array dimension whose size does not equal 1, %treating the elements as vectors. This dimension becomes 1 while the sizes of all other dimensions remain the same.
	TP = sum((predict == 1) & (real == 1));
	FP = sum((predict == 1) & (real == 0));
	FN = sum((predict == 0) & (real == 1));
	Recall = TP /(TP + FN);
	precision = TP /(TP + FP);
	f1_measure = (2* Recall * precision)/(Recall + precision);
	compute_measure = f1_measure;
end
%%

function plot_func = make_charts(number_of_selected_subsets, subsets_scores)
    figure(1)
    [m n]=size(subsets_scores)
    disp(subsets_scores)
    num1=number_of_selected_subsets-1
    %num2=subsets_scores(1:1:n-1)
    i=1;
    for ele=1:n
        if subsets_scores(ele)==-1
            subsets_scores_new(i)=0;
            i=i+1;
        else
            subsets_scores_new(i)=subsets_scores(ele);
            i=i+1;
        end
    end
    
    disp('num1')
    disp(num1)
    num2=subsets_scores_new(1:1:n-1);
    disp('num2')
    disp(num2)
    %disp(subsets_scores_new)
	x = [1:1:num1];
%     disp('*')
%     length(x)
%     length(num2)
	plot(x, num2,'r')
    grid on
    xlabel('Number of selected subsets')
    ylabel('F1 score')
    plot_func=1;
    figure(2)
    bar(x, num2,0.2,'r')
    xlabel('Number of selected subsets')
    ylabel('F1 score')
end