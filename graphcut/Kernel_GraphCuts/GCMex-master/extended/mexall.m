function mexall
%
% Compiling mex files for extended \alpha expansion
%

fprintf(1, 'If you have not yet configured your mex compiler, this is the right time.\n');
fprintf(1, 'Type:\n>> mex -setup\nand let Matlab detect installed compilers.\n');
fprintf(1, 'Under Linux you should select gcc compiler, for windows choose visual studio.\n\n');

if isunix
    mex -O -largeArrayDims CXXFLAGS="\$CXXFLAGS -Wno-write-strings"...
        QPBO.cpp QPBO_extra.cpp QPBO_maxflow.cpp QPBO_postprocessing.cpp...
        compact_a_expand_mex.cpp -output compact_a_expand_mex
else
    mex -O -largeArrayDims ...
        QPBO.cpp QPBO_extra.cpp QPBO_maxflow.cpp QPBO_postprocessing.cpp...
        compact_a_expand_mex.cpp -output compact_a_expand_mex
end

fprintf(1, 'If everything went smoothly type\n');
fprintf(1, '>>doc compact_a_expand_mex\nto see relevant documentation.\n\n');
