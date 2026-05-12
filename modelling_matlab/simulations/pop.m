function [newlist,element] = pop(list,index)
%newlist = pop(list) pop the last element out of an existing list
%   returns the list without the last element(s)
%
%newlist = pop(list,index) pulls out the indexed element
%   returns the list without the indexed element(s)
%
%[newlist,element] = pop(list) returns the indexed element also

if nargin < 2
    index = length(list);
end

element = list(index);
newlist = list(~ismember(list,element));

end

