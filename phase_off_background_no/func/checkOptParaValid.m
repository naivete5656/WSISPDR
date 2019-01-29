function valid=checkOptParaValid(optparas)

parameter=[optparas.w_smooth_spatio, optparas.w_sparsity, optparas.sel,...
    optparas.epsilon, optparas.gamma, optparas.m_scale, optparas.maxiter, optparas.tol];

if ~isempty(find(parameter<0, 1))
    valid=0;
else
    valid=1;
end