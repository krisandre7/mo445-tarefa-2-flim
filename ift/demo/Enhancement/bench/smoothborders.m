function imgout=smoothborders(workdir,extension);

if (workdir(end) ~= '/') workdir(end+1) = '/'; end
ext=extension(max(find(extension=='.'))+1:end);

files=dir([workdir extension]);
for(i=1:length(files))
    inputfile = [workdir files(i).name];
    fprintf(1,'%d/%d - %s\n',i,length(files),inputfile);
    if strcmp(ext,'fscn') 
        img = iftReadFImage(inputfile);
    elseif strcmp(ext,'pgm')
        img = imread(inputfile);
        img = single(double(img));
    else
        fprintf(1,'Error: invalid image format: %s\n',ext);
    end
    imgl = (img); % +1 from negative frontier of SVM 
    imgl = imgl ./ max(imgl(:));    
    wgb = imgl; % required by benchmark BSR
    
    outputfile = files(i).name;
    outputfile = [outputfile(1:min(find(outputfile=='.'))) 'mat'];
    outputfile = [workdir outputfile];
    save(outputfile,'wgb');
end