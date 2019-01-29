function kernel=getKernel(kernparas,ang,debug)
%% The corresponding point spread kernel function with phase retardation $angle$
radius=kernparas.radius;
R=kernparas.R;
W=kernparas.W;
zetap=kernparas.zetap;


[xx,yy] = meshgrid(-radius:radius,-radius:radius);
rr = sqrt(xx.^2 + yy.^2);

kernel1 = pi*R^2*somb(2*R*rr);     
kernel2 = pi*(R-W)^2*somb(2*(R-W)*rr);    
%kernel1 = kernel1/sum(abs(kernel1(:))); 
%kernel2 = kernel2/sum(abs(kernel2(:)));
kernelr = kernel1 - kernel2;
kerneli=(zetap*cos(ang)-sin(ang))*kernelr;

kernel=kerneli;
kernel(radius+1,radius+1)=kernel(radius+1,radius+1)+sin(ang);
kernel=kernel/norm(kernel); %for kernel normalization

if debug
    figure;
    surf(xx,yy,kernel);
    colormap jet
    
%     pause
end
