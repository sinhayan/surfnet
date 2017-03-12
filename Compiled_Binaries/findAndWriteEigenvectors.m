function findAndWriteEigenvectors(prefix, numeigv, m, n)
try 
    fid = fopen([prefix '.matrix.bin'], 'rb')   %open file
    nz = fread(fid, 1, 'int32');
    rows = fread(fid, nz, 'int32');      %read in the data
    cols = fread(fid, nz, 'int32');      %read in the data
    vals = fread(fid, nz, 'double');   %read in the data
    fclose(fid);                        %close file

    %rows = load([prefix '.rows.txt']);
    %cols = load([prefix '.cols.txt']);
    %vals = load([prefix '.vals.txt']);
    A = sparse(rows+1, cols+1, vals, m, n);
    OPTS.maxit = 5000;
    [V, D] = eigs(A, numeigv, 'LM', OPTS);
    S = diag(D);
    
    fid1 = fopen([prefix '.eigenvectors.bin'], 'wb');
    fwrite(fid1, full(abs(V)), 'double');
    fclose(fid1);
    
    fid2 = fopen([prefix '.eigenvalues.bin'], 'wb');
    fwrite(fid2, S, 'double');
    fclose(fid2);

    %dlmwrite([prefix '.eigenvectors.txt'], abs(V));
    %dlmwrite([prefix '.eigenvalues.txt'], S);
    exit
catch exception
    fprintf('Exception');
    exit
end
end