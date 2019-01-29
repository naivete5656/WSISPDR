function valid=checkKernParaValid(kernparas)

parameter=[kernparas.R,kernparas.W,kernparas.radius,...
    kernparas.zetap,kernparas.dicsize];

if ~isempty(find(parameter<=0, 1))
    valid=0;
else
    valid=1;
end

if parameter(1)<parameter(2)
    valid=0;
end
