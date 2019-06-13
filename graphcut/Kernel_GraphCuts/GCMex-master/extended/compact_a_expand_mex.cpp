#include "mex.h"

#include "QPBO.h"

#include <limits>
#include <string.h>

/*
 * \alpha expansion optimization for multi label non-submodular energy
 *
 * Usage: 
 *   [x e] = compact_a_expand_mex(UTerm, PTerm_i, PTerm_v, [itr], [ig]);
 *
 * Inputs:
 *  UTerm   - LxN matrix of unary terms (N-number of variables, L-number of labels)
 *            Each term (col) is a pair [Ei(0), Ei(1), Ei(2), ...].'
 *  PTerm_i - NxN sparse matrix of indices into pairwise terms 
 *            Each term (col i) lists its neighbors j's.
 *            Where i,j are the indices of the variables 
 *		      w_ij is index into PTerm_v
 *  PTerm_v - (L^2)x|vi| matrix of the different col-stack matrices V. (1 based index)
 *  itr     - number of iterations (optional)
 *  ig      - "hot start" 1xN matrix, with initial labeling.  (optional)
 *
 * Outputs:
 *	x	  - double L vectr of final labels
 *	e	  -	[Energy Eunary Epair-wise nIter]
 *
 * 
 * Input formats:
 *  PTerm_i - sparse NxN matrix 
 *                                j \ i ... From ...
 *                                   +----------------
 *                                .  |
 *                                .  |
 *                                .  |
 *                                T  |
 *                                o  |     w_ij
 *                                .  |
 *                                .  |
 *                                .  |
 *
 *
 *  A matrix V (to be pack in col-stack into PTerm_v)
 *                              l_i \ l_j ... To ...
 *                                   +----------------
 *                                .  |
 *                                .  |
 *                                .  |
 *                             From  |  E_{ij}(l_i,l_j) = V[l_i + l_j*L]
 *                                .  |
 *                                .  |
 *                                .  |
 *
 *  For lighting energy the matrix V should be constructed as
 *  >> Vh = abs( bsxfun( @minus, sL(:,1)+sL(:,2), sL(:,1)') ) + gamma * abs( bsxfun( @minus, sL(:,2), sL(:,2)' ) );
 *  >> Vv = abs( bsxfun( @minus, sL(:,1)+sL(:,3), sL(:,1)') ) + gamma * abs( bsxfun( @minus, sL(:,3), sL(:,3)' ) );
 *
 * compiling using:
 * >> mexall
 *
 *
 *
 *
 *  This wrapper for Matlab was written by Shai Bagon (shaibagon@gmail.com).
 *  Department of Computer Science and Applied Mathmatics
 *  Wiezmann Institute of Science
 *  http://www.wisdom.weizmann.ac.il/
 *
 *
 *  The \alpha-expansion move is executed using QPBO construct by Valdimir
 *  Kolmogorov
 *	(available from http://pub.ist.ac.at/~vnk/software.html#QPBO):
 *
 *  [1] Optimizing binary MRFs via extended roof duality
 *       C. Rother, V. Kolmogorov, V. Lempitsky, and M. Szummer.
 *       CVPR'2007.
 *
 *  [2] Efficient Approximate Energy Minimization via Graph Cuts
 *       Yuri Boykov, Olga Veksler, Ramin Zabih,
 *       IEEE transactions on PAMI, vol. 20, no. 12, p. 1222-1239, November
 *       2001.
 * 
 *  [3] Matlab Wrapper for Graph Cut.
 *       Shai Bagon.
 *       in https://github.com/shaibagon/GCMex, December 2011.
 *
 *  This software can be used only for research purposes, you should  cite ALL of
 *  the aforementioned papers in any resulting publication.
 *  If you wish to use this software (or the algorithms described in the
 *  aforementioned paper)
 *  for commercial purposes, you should be aware that there is a US patent:
 *
 *      R. Zabih, Y. Boykov, O. Veksler,
 *      "System and method for fast approximate energy minimization via
 *      graph cuts ",
 *      United Stated Patent 6,744,923, June 1, 2004
 *
 *
 *  The Software is provided "as is", without warranty of any kind.
 *
 */

inline
void null_fcn(...) {}

#ifdef DEBUG
#define DEBUGmexPrintf mexPrintf
#else
#define DEBUGmexPrintf null_fcn  
#endif

// inputs
enum {
    iU  = 0,
    iPI,
	iPV,
    iT,
	iS,
    nI
};

// outputs
enum {
    oX = 0,
    oE,
    nO
};

void my_err_function(char* msg) {
    mexErrMsgTxt(msg);
}


template <typename T, typename REAL>
void compact_a_expand(int nout, mxArray* pout[], int nin, const mxArray* pin[]);

template<class T, class REAL, class Label>
REAL Energy(const T* pU, const double*  pVi, const mwIndex* pir, const mwIndex* pjc, const T* pV, const Label* pL, unsigned int N, unsigned int L, REAL* pdE = NULL, REAL* psE = NULL);

void
mexFunction(
    int nout,
    mxArray* pout[],
    int nin,
    const mxArray* pin[])
{
    if (nin < iT || nin > nI)
        mexErrMsgIdAndTxt("compact_a_expand_mex:nin","Expecting %d inputs", nI);
    
    if (nout == 0)
        return;
    if (nout > nO)
         mexErrMsgIdAndTxt("compact_a_expand_mex:nout", "Expecting %d outputs", nO);
    
    if ( mxIsComplex(pin[iU]) || mxIsSparse(pin[iU]) || !mxIsDouble(pin[iU]) )
        mexErrMsgIdAndTxt("compact_a_expand_mex:unary_term",
                "Unary term must be full real matrix");
    
	if ( mxIsComplex(pin[iPI]) || !mxIsSparse(pin[iPI]) || !mxIsDouble(pin[iPI]) )
        mexErrMsgIdAndTxt("compact_a_expand_mex:pairwise_term",
                "Pair-wise index term must be sparse real matrix");

    if ( mxIsComplex(pin[iPV]) || mxIsSparse(pin[iPV]) || !mxIsDouble(pin[iPV]) )
        mexErrMsgIdAndTxt("compact_a_expand_mex:pairwise_term",
                "Pair-wise index term must be full real matrix");
    
    
    if ( mxGetClassID(pin[iPV]) != mxGetClassID(pin[iU]) )
        mexErrMsgIdAndTxt("compact_a_expand_mex:energy_terms",
                "Both energy terms must be of the same class");
        
    mwSize L = mxGetM(pin[iU]);
    if ( mxGetNumberOfDimensions(pin[iU]) != 2 ||  mxGetM(pin[iU]) != L )
        mexErrMsgIdAndTxt("compact_a_expand_mex:unary_term_size",
                "Unary term must be LxN matrix");
    
	if ( mxGetNumberOfDimensions(pin[iPI]) != 2 || mxGetM(pin[iPI]) != mxGetN(pin[iU]) || mxGetN(pin[iPI]) != mxGetN(pin[iU]) )
        mexErrMsgIdAndTxt("compact_a_expand_mex:piarwise_index_term_size",
                "pair-wise index term must be NxN sparse matrix");
    
	if ( mxGetNumberOfDimensions(pin[iPV]) != 2 || mxGetM(pin[iPV]) != L*L )
        mexErrMsgIdAndTxt("compact_a_expand_mex:piarwise_term_size",
                "pair-wise V term must be L^2 x |vi| matrix");
    

    switch (mxGetClassID(pin[iU])) {
        case mxDOUBLE_CLASS:  
			// only double supported for time being
            return compact_a_expand<double, double>(nout, pout, nin, pin);
        case mxINT8_CLASS:
        case mxCHAR_CLASS:
//            return compact_a_expand<char, int>(nout, pout, nin, pin);
        case mxSINGLE_CLASS:
//            return compact_a_expand<float, float>(nout, pout, nin, pin);
        case mxUINT8_CLASS:
//            return compact_a_expand<unsigned char, int>(nout, pout, nin, pin);
        case mxINT16_CLASS:
//            return compact_a_expand<short, int>(nout, pout, nin, pin);
        case mxUINT16_CLASS:
//            return compact_a_expand<unsigned short, int>(nout, pout, nin, pin);
        case mxINT32_CLASS:
//            return compact_a_expand<int, int>(nout, pout, nin, pin);
        case mxUINT32_CLASS:
//            return compact_a_expand<unsigned int, int>(nout, pout, nin, pin);
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
        default:
            mexErrMsgIdAndTxt("compact_a_expand_mex:energy_class",
                    "Unknown/unsupported class %s",mxGetClassName(pin[iU]));
    } 
    return;
}

template <typename T, typename REAL>
void compact_a_expand(int nout, mxArray* pout[], int nin, const mxArray* pin[])
{ 
    typedef typename QPBO<REAL>::NodeId NodeId;
    typedef typename QPBO<REAL>::EdgeId EdgeId;
    typedef typename QPBO<REAL>::ProbeOptions ProbeOptions;
	
	typedef unsigned int Label;

    mwSize N  = mxGetN(pin[iU]); // number of nodes/variables
    mwSize L  = mxGetM(pin[iU]); // number of labels    
    mwSize nV = mxGetN(pin[iPV]); // number of different V matrices


	int nIter = std::numeric_limits<int>::max(); // default value
	ProbeOptions po;

	Label* CurrentLabels = new Label[N];
	memset( CurrentLabels, 0, N*sizeof(Label) ); // default labels
	
	Label* pL = new Label[N];
	memset( CurrentLabels, 0, N*sizeof(Label) ); // default labels

	mwIndex* eI = new mwIndex[N]; // effective index into reduced graph
	double * pig = 0; // ig if given

#ifdef PROBE
	int* pMap = new int[N]; // for "probe"
#endif

	// optional parameters :
	//  iT - number of iterations
	//  iS - init labeling
	switch (nin) {
	case iT:
		DEBUGmexPrintf("DEBUG no optional params given\n");
		// no optional parameters were given - retain default values		
		break;
	case iS:
		DEBUGmexPrintf("DEBUG single optional params given\n");
		// a single optinal parameter was given
		if ( !mxIsComplex(pin[iT]) && !mxIsSparse(pin[iT]) && mxGetNumberOfElements(pin[iT])==1) {
			// the parameter is nIter
			nIter = static_cast<int>(mxGetScalar(pin[iT]));
		} else if ( !mxIsComplex(pin[iT]) && !mxIsSparse(pin[iT]) && mxGetNumberOfElements(pin[iT]) == N && mxIsDouble(pin[iT]) ) {
			// the parameter is ig
			pig = mxGetPr(pin[iT]);
			for ( mwIndex ii(0); ii < N ; ii++ ) {
				Label li = static_cast<Label>(pig[ii]-1); // convert from 1-based labels to 0-based
				if ( li < 1 || li >= L )
					mexErrMsgIdAndTxt("compact_a_expand_mex:ig", "label value out of bound");
				CurrentLabels[ii] = li;
			}
		} else {
			 mexErrMsgIdAndTxt("compact_a_expand_mex:optional_params", "invalid optional params: should be num of iter and ig");
		}
		break;
	case nI:
		DEBUGmexPrintf("DEBUG both optional params given\n");
		// both given
		if ( mxIsComplex(pin[iT]) || mxIsSparse(pin[iT]) || mxGetNumberOfElements(pin[iT])!=1) 
			mexErrMsgIdAndTxt("compact_a_expand_mex:number_of_iter","number of iterations must be real scalar");

		if ( mxIsComplex(pin[iS]) || mxIsSparse(pin[iS]) || mxGetNumberOfElements(pin[iS])!=N || !mxIsDouble(pin[iS])) 
			mexErrMsgIdAndTxt("compact_a_expand_mex:ig","ig must be real vector of length %d", N);

		// get number of iterations
		nIter = static_cast<int>(mxGetScalar(pin[iT]));

		// get the init labels
		pig = mxGetPr(pin[iS]);
		for ( mwIndex ii(0); ii < N ; ii++ ) {
			Label li = static_cast<Label>(pig[ii]-1); // convert from 1-based labels to 0-based
			if ( li < 0 || li >= L )
				mexErrMsgIdAndTxt("compact_a_expand_mex:ig", "label value out of bound");
			CurrentLabels[ii] = li;
		}
		break;
	default:
		mexErrMsgIdAndTxt("compact_a_expand_mex:num_of_inputs", "unclear error");
	}

  
    T* pU  = (T*)mxGetData(pin[iU]);
    T* pV  = (T*)mxGetData(pin[iPV]);
	
	double*  pVi = mxGetPr(pin[iPI]);
	mwIndex* pir = mxGetIr(pin[iPI]);
	mwIndex* pjc = mxGetJc(pin[iPI]);

    
	
	unsigned int itr(0);
	bool done = (itr >= nIter);	
	
	
	REAL dE, sE, E(0), cE = Energy<T, REAL, Label>(pU, pVi, pir, pjc, pV, CurrentLabels, N, L, &dE, &sE);	

	// start alpha-expand - loop over labels
	while ( ! done ) {
		done = true;
		itr++;
		

		// increment "alpha"
		for ( Label alpha(0); alpha < L ; alpha++ ) {			
			

			memset( eI, 0, N*sizeof(mwIndex) );

			// in the binary sub-problem:
			//         1,  l_i = alpha
			// x_i =
			//         0,  l_i = CurrentLabels[ii] (retain)


			mwSize eN = 0; // effective number of nodes		
			for ( mwSize ii=0; ii < N ; ii++ ) {
				if ( CurrentLabels[ii] == alpha )
					continue;

				eI[ii] = eN;
				eN++; 
			}

			if ( eN < 1 )
				continue;

			DEBUGmexPrintf("DEBUG alpha = %d, effective number of variables for expand step = %d\n", alpha+1, eN);

			// construct QPBO
			// note that N and E are only estimates for mem allocation
			QPBO<REAL>* qpbo = new QPBO<REAL>(eN, 6*eN*nV, my_err_function); // construct with an error message function

			qpbo->AddNode(eN);

			// add unary and pair-wise terms
			for ( mwSize ii=0; ii < N ; ii++ ) {
				if ( CurrentLabels[ii] == alpha )
					continue;

				REAL E0(0), E1(0);

				E0 = pU[ii*L + CurrentLabels[ii]];
				E1 = pU[ii*L + alpha];

				// ii col index (from), jj row index jj<-pir[ri] (to)
				for (mwIndex ri = pjc[ii] ; // starting row index
					ri < pjc[ii+1]  ; // stopping row index
					ri++)  {
						mwIndex jj = pir[ri];
						mwIndex vi = static_cast<mwIndex>(pVi[ri]) - 1;
						if ( vi < 0 || vi >= nV )
							mexErrMsgIdAndTxt("compact_a_expand_mex:v_index_exceeds_dims", "index into matrices V out of bound");
						
						T * Vij = pV + L*L*vi;

						if ( CurrentLabels[jj] == alpha ) {
							// "fold" this edge into the unary
							E0 += Vij[L*CurrentLabels[ii] + alpha];
							E1 += Vij[L*alpha             + alpha];
						} else {
							// Eij(x_i, x_j)
							qpbo->AddPairwiseTerm( static_cast<NodeId>( eI[ii] ), static_cast<NodeId>( eI[jj] ),
								static_cast<REAL>(Vij[L*CurrentLabels[ii] + CurrentLabels[jj]]), // E00
								static_cast<REAL>(Vij[L*CurrentLabels[ii] + alpha            ]), // E01
								static_cast<REAL>(Vij[L*alpha             + CurrentLabels[jj]]), // E10
								static_cast<REAL>(Vij[L*alpha             + alpha            ])  // E11
								);
						}
				}

				qpbo->AddUnaryTerm(eI[ii], E0, E1);

			} // done adding nodes and edges
			
			// in case where duplicate edges were insderted
			qpbo->MergeParallelEdges();

			DEBUGmexPrintf("DEBUG alpha = %d, done constructing binary sub-problem\n", alpha+1);

			// solve (regular qpbo)
			qpbo->Solve();
			qpbo->ComputeWeakPersistencies();

			DEBUGmexPrintf("DEBUG alpha = %d, done QPBO optimiztion\n", alpha+1);
#ifdef PROBE
			// "probe"
			memset( pMap, 0, N*sizeof(int) );

			po.directed_constraints = 1; // 1: all possible directed constraints are added, if there is sufficient space for edges (as specified by edge_num_max; see SetEdgeNumMax() function)
			po.weak_persistencies = 1; // 1: use weak persistency in the main loop (but not for probing operations)
			po.order_seed = 1; // otherwise: random permutation with random seed 'order_seed' is used.

			qpbo->Probe(pMap, po);

			DEBUGmexPrintf("DEBUG alpha = %d, done probe w/o weakpersistency \n", alpha+1);

			qpbo->ComputeWeakPersistencies();

			DEBUGmexPrintf("DEBUG alpha = %d, done QPBO - \"Probe\" optimiztion\n", alpha+1);
#endif

			

			memcpy(pL, CurrentLabels, N*sizeof(Label));

#ifdef DEBUG
			int unl(0); // count number of unlabeled nodes
#endif
			int xi;
			// get the binary labels of the sub-problem
			for (mwSize ii(0); ii < N ; ii++ ) {
				if ( pL[ii] == alpha )
					continue;
#ifdef PROBE
				xi = qpbo->GetLabel(pMap[eI[ii]]/2) + pMap[eI[ii]]%2;
#else
				xi = qpbo->GetLabel(eI[ii]);
#endif
				if (xi==1) {
					pL[ii] = alpha;
#ifdef DEBUG
				} else {
					unl += (xi<0);			
#endif
				} // else (either xi=0 or xi<0 we do not change the labels)

			}
#ifdef DEBUG
			DEBUGmexPrintf("DEBUG alpha = %d, labeled %d out of %d (%.2f%%)\n", alpha+1, eN-unl, eN, (100.0*(eN-unl))/double(eN));
#endif

			// deallocate			
			delete qpbo;

			// do we accept this labeling?
			E = Energy<T, REAL, Label>(pU, pVi, pir, pjc, pV, pL, N, L, &dE, &sE);
			if ( E < cE ) {
				memcpy(CurrentLabels, pL, N*sizeof(Label));
				done = false;
				cE = E;
			}
			
		
		} // done expanding alpha

		
		
		// are we done?
		done = done || ( itr >= nIter );				
	} // main loop over labels "alpha"




	// allocate output
	pout[oX] = mxCreateDoubleMatrix(N, 1, mxREAL);
	double* pX = mxGetPr(pout[oX]);

	// read solution
	for ( mwSize ii=0; ii < N ; ii++ ) {
		pX[ii] = static_cast<double>( CurrentLabels[ii] + 1 ); // from 0-based labels to 1-based labels.
	}

	// final energy
	E = Energy<T, REAL, Label>(pU, pVi, pir, pjc, pV, CurrentLabels, N, L, &dE, &sE);

	pout[oE] = mxCreateDoubleMatrix(1, 4, mxREAL);
	double* pE = mxGetPr(pout[oE]);
	pE[3] = itr;
	pE[0] = E;
	pE[1] = dE;
	pE[2] = sE;

#ifdef PROBE
	delete[] pMap;
#endif

	delete[] CurrentLabels;
	delete[] pL;
	delete[] eI;
}

/*
 * compute energy (objective value)
 * for 0-based labels
 */
template<class T, class REAL, class Label>
inline
REAL Energy(
	const T* pU, 
	const double*  pVi, const mwIndex* pir, const mwIndex* pjc, const T* pV, 
	const Label* pL, unsigned int N, unsigned int L, REAL* pdE, REAL* psE)
{
	REAL dE(0), sE(0);
	unsigned int L2 = L*L;

	for ( mwIndex ii(0) ; ii < N ; ii++ ) {
		// add unary
		dE += pU[ L*ii + pL[ii] ];

		// add pair-wise terms
		
		// ii col index (from), jj row index jj<-pir[ri] (to)
		for (mwIndex ri = pjc[ii] ; // starting row index
			ri < pjc[ii+1]  ; // stopping row index
			ri++)  {
				mwIndex jj = pir[ri];
				mwIndex vi = static_cast<mwIndex>(pVi[ri]) - 1;

				sE += *(pV + L2*vi + pL[ii] + pL[jj]*L);
		}

	}
	DEBUGmexPrintf("DEBUG energy = %.2f + %.2f = %.2f\n", dE, sE, dE+sE);
	if ( pdE != NULL )
		pdE[0] = dE;
	if ( psE != NULL )
		psE[0] = sE;

	return sE+dE;
}

