function parsave(fname, var)
    var_name = inputname(2);
    S.(var_name) = var;
   
    % Save the fields of the struct as individual variables in the .mat file
    save(fname, '-struct', 'S');
end