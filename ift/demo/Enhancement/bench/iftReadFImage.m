function img=iftReadFImage(filename)
img = zeros(1,1);
pF=fopen(filename,'r');
if (pF == -1)
    fprintf(1,sprintf('Error while reading file %s\n',filename));
    return;
end
[type,count]=fscanf(pF,'%s\n',1);
if (count ~= 1) fprintf(1,'Reading error\n'); end
if (strcmp(type,'FSCN'))
    [size,count]=fscanf(pF,'%d %d %d\n',3);
    if (count ~= 3) fprintf(1,'Reading error\n'); end
    [d,count]=fscanf(pF,'%f %f %f\n',3);
    if (count ~= 3) fprintf(1,'Reading error\n'); end  
    img=zeros(size');
    n=prod(size);
    [buf,count]=fread(pF,n,'single');
    if (count ~= n) fprintf(1,'Reading error\n'); end
    img=reshape(buf,size(1),size(2),size(3))';
    fclose(pF);
else
    fprintf(1,'Invalid file type\n');
end

end