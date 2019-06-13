// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.



/*****************************************
 *
 * Matlab mex file for computing the geodesic distance
 * Matlab Syntax: (used to call this mex file)
 *	[label_matrix, distance_matrix] = labeled_geodesic_dist_mex(marker_matrix, mask_matrix);
 *		Inputs
 *       	-marker_matrix: the labeled elements that the geodesic distance is calculated to
 *       	-mask_matrix: the logical matrix(mask) containing the valid traversal locations
 *		Outputs
 *       	-label_matrix: the label of the nearest marker objects
 *			-distance_matrix: the geodesic distance to the nearest nonzero labeled element
 *
 ******************************************/

#include "mex.h"

typedef struct stack {
	int *data;
	int nb_el;
	int size;
} Stack;

Stack Stack_Init(int new_size);
void push(Stack *S, int val);
int pop(Stack *S);
void Destroy_Stack(Stack S);

void geo_dist(int *marker_matrix, double *dist_matrix, const mxLogical *mask_matrix, const int m, const int n);


/* Matlab Interface Function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Declare local variables */
	double *dist_matrix;
	mxLogical *mask_matrix;
	int *marker_matrix;
	int m, n;
	register int k;

	/* pointers to account for multiple input formats */
	double *db_ptr;
	float *fl_ptr;
	int *int_ptr;
	unsigned int *uint_ptr;
	short int *sint_ptr;
	unsigned short int *usint_ptr;
	char *c_ptr;
	unsigned char *uc_ptr;

	/***********************
	 * Check I/O arguments *
	 ***********************/
	/* check that there 2 or 3 input arguments */
	if (nrhs != 2)
		mexErrMsgTxt("Invalid number input arguments.\n Usage:  [label_matrix, distance_matrix] = labeled_geodesic_dist_mex(marker_matrix, mask_matrix)");
	/* check there are 1 or 2 output arguments */
	if (nlhs > 2 || nlhs < 1)
		mexErrMsgTxt("Invalid number output arguments.\n Usage:  [label_matrix, distance_matrix] = labeled_geodesic_dist_mex(marker_matrix, mask_matrix)");
	/* Check input type for marker matrix */
	if (mxIsComplex(prhs[0]) || mxIsEmpty(prhs[0]) || mxIsSparse(prhs[0]))
		mexErrMsgTxt("Invalid input argument:\n  marker_matrix: must be real, non-empty, non-sparse matrix.\n)");
	/* check input type for mask matrix */
	if (!mxIsLogical(prhs[1]) || mxIsComplex(prhs[1]) || mxIsEmpty(prhs[1]) || mxIsSparse(prhs[1]))
		mexErrMsgTxt("Invalid input argument:\n  Useage:  [label_matrix, distance_matrix] = labeled_geodesic_dist_mex(marker_matrix, <logical>)");

	/* save off matrix dimensions to local variables */
	m = (int) mxGetM(prhs[0]);
	n = (int) mxGetN(prhs[0]);

	/* Check that the dimensions of the 2 input matrices are equal */
	if (m != (int)mxGetM(prhs[1]) || n != (int)mxGetN(prhs[1]))
		mexErrMsgTxt("Input matrix dimension mismatch.");
	if (n < 1 || m < 1)
		mexErrMsgTxt("Empty input matrix.");

	/* generates and initializes a matrix of zeros */
	plhs[0] = mxCreateNumericMatrix(m, n, mxINT32_CLASS, mxREAL);
	/* assign output pointer to the newly created output matrix */
	marker_matrix = mxGetData(plhs[0]);

	/* copy over the input image into output memory with typecast */
	switch (mxGetClassID(prhs[0])) {
	case mxDOUBLE_CLASS:
		db_ptr = mxGetPr(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)db_ptr[k];
		}
		break;
	case mxSINGLE_CLASS:
		fl_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)fl_ptr[k];
		}
		break;
	case mxINT32_CLASS:
		int_ptr = mxGetData(prhs[0]);
		memcpy(marker_matrix, int_ptr, m * n * sizeof(int));
		break;
	case mxUINT32_CLASS:
		uint_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)uint_ptr[k];
		}
		break;
	case mxINT16_CLASS:
		sint_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)sint_ptr[k];
		}
		break;
	case mxUINT16_CLASS:
		usint_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)usint_ptr[k];
		}
		break;
	case mxINT8_CLASS:
		c_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)c_ptr[k];
		}
		break;
	case mxUINT8_CLASS:
		uc_ptr = mxGetData(prhs[0]);
		for (k = 0; k < m * n; k++) {
			marker_matrix[k] = (int)uc_ptr[k];
		}
		break;
	default:
		mexErrMsgTxt("Invalid input image type");
	}

	/* If the user has specified 2 outputs, create the distance matrix */
	if (nlhs == 2) {
		/* generates and initializes a Matlab double matrix of zeros */
		plhs[1] = mxCreateDoubleMatrix(m, n, mxREAL);
		dist_matrix = mxGetPr(plhs[1]);
	} else {
		/* The user does not want the distance matrix, so don't assign it to the output array */
		/*dist_matrix = (double*)malloc(m*n*sizeof(double)); */
		dist_matrix = NULL;
	}

	/* get pointer to the mask matrix */
	mask_matrix = mxGetLogicals(prhs[1]);

	/*************************
	 * Run geodesic distance worker function
	 * Inputs:
	 * marker_matrix: the marker matrix at the start of the distance calculation
	 * dist_matrix: empty (initialized to zeros) matrix, or a NULL pointer if distances are not requested
	 * mask_matrix: the matrix containing the elements that need to be assigned labels
	 * [m n] = size(matrix_out): the matrix dimensions (m = numbers rows) and (n = number columns)
	 *************************/
	geo_dist(marker_matrix, dist_matrix, mask_matrix, m, n);
	/*************************
	 * Outputs:
	 * marker_matrix: holds the label of the closest object found by geodesic dilation
	 * dist_matrix: holds the geodesic distance of any located element, infinity if a element was not found
	 *************************/

}


void geo_dist(int *marker_matrix, double *dist_matrix, const mxLogical *mask_matrix, const int m, const int n) {
	/* define local variables */
	int i, j, element_assigned_flag, iteration_count, nb_elements_in_matrix, end_indx;
	register int k;
	double inf_val, nan_val;
	Stack loc_stack; /* int stack to hold the locations */
	Stack label_stack; /* int stack to hold the locations */

	nb_elements_in_matrix = (int)m * n;

	if (dist_matrix != NULL) {
		/* initialize distance matrix to infinity */
		inf_val = mxGetInf(); /* --> returns IEEE infinity value */
		nan_val = mxGetNaN(); /* --> returns IEEE NaN value */
		for (k = 0; k < nb_elements_in_matrix; k++) {
			if (marker_matrix[k] == 0) { /* only write inf to locations not under a marker */
				if (mask_matrix[k] != 0) {
					dist_matrix[k] = inf_val;
				} else {
					dist_matrix[k] = nan_val;
				}
			}
		}
	}


	loc_stack = Stack_Init(m);
	label_stack = Stack_Init(m);

	iteration_count = 0; /* counter to hold the geodesic distance of the current iteration */
	element_assigned_flag = 1; /* flag to control the stopping condition */
	/* stop condition: when a loop over the list of elements that need a label produces no new labels */
	while (element_assigned_flag == 1) {
		element_assigned_flag = 0;
		iteration_count++;


		/* loop over the pixels */
		for (k = 0; k < nb_elements_in_matrix; k++) {
			if (marker_matrix[k] == 0 && mask_matrix[k] != 0 ) {
				i = k % m;
				j = k / m;


				/* left */
				if (j > 0 && marker_matrix[k - m] != 0) {
					push(&loc_stack, k);
					push(&label_stack, marker_matrix[k - m]);
				}
				/* top */
				if (i > 0 && marker_matrix[k - 1] != 0) {
					push(&loc_stack, k);
					push(&label_stack, marker_matrix[k - 1]);
				}
				/* bottom */
				if (i < m - 1 && marker_matrix[k + 1] != 0) {
					push(&loc_stack, k);
					push(&label_stack, marker_matrix[k + 1]);
				}
				/* right */
				if (j < n - 1 && marker_matrix[k + m] != 0) {
					push(&loc_stack, k);
					push(&label_stack, marker_matrix[k + m]);
				}
			}
		}


		/* commit label updates to marker matrix */
		if (loc_stack.nb_el > 0) {
			/* set the flag that shows a element was labeled at this distance */
			element_assigned_flag = 1;
			/* commit the changes to marker_matrix */
			if (dist_matrix == NULL) {
				while (loc_stack.nb_el > 0) {
					marker_matrix[pop(&loc_stack)] = pop(&label_stack);
				}
			} else {
				while (loc_stack.nb_el > 0) {
					k = pop(&loc_stack);
					marker_matrix[k] = pop(&label_stack);
					dist_matrix[k] = (double)iteration_count;
				}
			}
		}

	} /* end while loop */

	/* free dynamic memory */
	Destroy_Stack(loc_stack);
	Destroy_Stack(label_stack);
} /* end geo_dist */




Stack Stack_Init(int new_size) {
	Stack S;
	S.size = new_size;
	S.data = mxCalloc(S.size, sizeof(int));
	if (S.data == NULL)
		mexErrMsgTxt("Error: out of heap space.\n");
	S.nb_el = 0;
	return S;
}

void push(Stack *S, int val) {
	if (S->nb_el >= S->size) {
		/* if the stack is out of space, double its size */
		S->size = (int)2 * (S->size);
		S->data = mxRealloc(S->data, sizeof(int) * S->size);
		if (S->data == NULL)
			mexErrMsgTxt("Error: out of heap space.\n");
	}
	S->data[(S->nb_el)++] = val;
}

int pop(Stack *S) {
	/* update the nb of elements in the stack, pop the last one */
	return S->data[--(S->nb_el)];
}

void Destroy_Stack(Stack S) {
	mxFree(S.data);
}



