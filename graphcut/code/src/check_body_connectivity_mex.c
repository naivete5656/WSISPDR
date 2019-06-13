// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.
        

/*****************************************
 *
 * Matlab mex file for computing the geodesic distance
 * Matlab Syntax: (used to call this mex file)
 *	image_out = check_body_connectivity_mex(image_in);
 *		Inputs
 *       	-image_in: the labeled image
 *		Outputs
 *       	-image_out: the labeled image where non connected bodies with the same label have been removed
 *
 ******************************************/

#include "mex.h"

#define calloc mxCalloc
#define malloc mxMalloc
#define free mxFree
#define realloc mxRealloc
#define printf mexPrintf

typedef struct stack {
    int *data;
    int nb_el;
    int size;
} Stack;

Stack Stack_Init(int new_size);
void push(Stack *S, int val);
int pop(Stack *S);
void Destroy_Stack(Stack S);

/* compare function used to quicksort the array */
int compare (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
}
int ij_to_lin(int i, int j, int m) {
    return j*m + i;
}

/*  image_out = check_body_connectivity(image_in) */
void generate_sub_mask(mxLogical *mini_mask, const int *image_in, int i_min, int i_max, int j_min, int j_max, int m, int obj_nb);
void generate_sub_image(int *mini_mask, const int *image_in, int i_min, int i_max, int j_min, int j_max, int m);
int* unique(const int *list_in, int *nb_elements, int modulo);
void image_features(int *edgeImage, const int nb_cells, const int *cell_nbs, int *bounding_box, const int *imageIn, const int m, const int n);


/* Matlab Interface Function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    /* Declare local variables */
    double *labeled_mini_image;
    int *image_in, *image_out, *mini_edge_image, *object_size, *body_neighbors, *max_neighbors, *neighbor_winner;
    int *edge_image, *cell_nbs, *bounding_box;
    mxLogical *mask;
    mxArray *mask_array, *connectivity, *bwlabel_call[2], *bwlabel_return[2];
    int m, n, highest_obj_nb, nb_objects, obj_nb, bwlabel_return_status;
    register int k;
    int nb_cells, max_val, winner_body, i, j, pixel, mini_m, mini_n, indx;
    int i_min, i_max, j_min, j_max;
    double temp;
    div_t divresult;
    
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
    /* check that there 2 input arguments */
    if(nrhs != 1 && nrhs != 2)
        mexErrMsgTxt("Invalid number input arguments.\n Usage:  [image_out] = check_body_connectivity(image_in)");
    /* check there are 1 or 2 output arguments */
    if(nlhs != 1)
        mexErrMsgTxt("Invalid number output arguments.\n Usage:  [image_out] = check_body_connectivity(image_in)");
    /* Check input type for marker matrix */
    if(mxIsComplex(prhs[0]) || mxIsEmpty(prhs[0]) || mxIsSparse(prhs[0]))
        mexErrMsgTxt("Invalid input argument:\n  marker_matrix: must be real, non-empty, non-sparse matrix.\n)");
    
    /* save off matrix dimensions to local variables */
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    
    if(n < 1 || m < 1)
        mexErrMsgTxt("Empty input matrix.");
    
    /* allocate memory for the local copy of the input image */
    image_in = calloc(m*n, sizeof(int));
    if(image_in == NULL)
        mexErrMsgTxt("Error: out of heap space.");
    
    /* copy over the input image into output memory with typecast */
    switch(mxGetClassID(prhs[0])) {
        case mxDOUBLE_CLASS:
            db_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)db_ptr[k];
            }
            break;
        case mxSINGLE_CLASS:
            fl_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)fl_ptr[k];
            }
            break;
        case mxINT32_CLASS:
            int_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)int_ptr[k];
            }
            break;
        case mxUINT32_CLASS:
            uint_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)uint_ptr[k];
            }
            break;
        case mxINT16_CLASS:
            sint_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)sint_ptr[k];
            }
            break;
        case mxUINT16_CLASS:
            usint_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)usint_ptr[k];
            }
            break;
        case mxINT8_CLASS:
            c_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)c_ptr[k];
            }
            break;
        case mxUINT8_CLASS:
            uc_ptr = mxGetData(prhs[0]);
            for(k = 0; k < m*n; k++) {
                image_in[k] = (int)uc_ptr[k];
            }
            break;
        default:
            mexErrMsgTxt("Invalid input image type");
    }
    
    connectivity = mxCreateDoubleScalar(4);
    
    /* generates and initializes a matrix of zeros */
    plhs[0] = mxCreateNumericMatrix(m, n, mxINT32_CLASS, mxREAL);
    /* assign output pointer to the newly created output matrix */
    image_out = mxGetData(plhs[0]);
    if(image_out == NULL)
        mexErrMsgTxt("Error: out of heap space.\n");
    /* init the output image */
    memcpy(image_out, image_in, m*n*sizeof(int));
    
    /* init the nb of objects to the number of elements in the input matrix */
    nb_cells = m*n;
    /* returns a list of nonzero elements in the input matrix (unsorted list) */
    cell_nbs = unique(image_in, &nb_cells, 113);
    /* check that there were nonzero elements in the input matrix */
    if(cell_nbs == NULL) {
        /* no objects in image_in */
        /* free memory then reutrn */
        mxDestroyArray(connectivity);
        free(image_in);
        return;
    }
    
    highest_obj_nb = cell_nbs[nb_cells-1];
    
    /* setup and call image features to extract bounding boxes and edge image */
    edge_image = calloc(m*n, sizeof(int));
    bounding_box = calloc(highest_obj_nb*4, sizeof(int));
    /* call image features to extract the basic data */
    image_features(edge_image, nb_cells, cell_nbs, bounding_box, image_in, m, n);
    
    highest_obj_nb++;
    /* loop over the currently labeled objects in the image */
    for(obj_nb = 1; obj_nb < highest_obj_nb; obj_nb++) {
        
        /* find the index in image feature data that corresponds to this object */
        indx = -1;
        for(k = 0; k < nb_cells; k++) {
            if(obj_nb == cell_nbs[k]) {
                indx = k;
                break;
            }
        }
        if(indx == -1) { /* this object does not really exist in the image */
            continue;
        }
        
        i_min = bounding_box[ij_to_lin(indx, 0, nb_cells)] - 1;
		if(i_min < 0)
			i_min = 0;
        i_max = bounding_box[ij_to_lin(indx, 1, nb_cells)] + 1;
		if(i_max >= m)
			i_max = m-1;
        j_min = bounding_box[ij_to_lin(indx, 2, nb_cells)] - 1;
		if(j_min < 0)
			j_min = 0;
        j_max = bounding_box[ij_to_lin(indx, 3, nb_cells)] + 1;
		if(j_max >= n)
			j_max = n-1;
		mini_m = i_max - i_min + 1;
        mini_n = j_max - j_min + 1;
        
        mask_array = mxCreateLogicalMatrix(mini_m, mini_n);
        mask = mxGetData(mask_array);
        if(mask == NULL)
            mexErrMsgTxt("Error: Out of heap space.\n");
        
        /* generate a sub image from the bounding boxes of the current object */
        generate_sub_mask(mask, image_in, i_min, i_max, j_min, j_max, m, obj_nb);
        
        /* call Matlab's bwlabel */
        bwlabel_call[0] = mask_array;
        bwlabel_call[1] = connectivity;
        /* [L, num] = bwlabel(BW, n) */
        bwlabel_return_status = mexCallMATLAB(2, bwlabel_return, 2, bwlabel_call, "bwlabel");
        if(bwlabel_return_status != 0)
            mexErrMsgTxt("Error occurred in Matlab bwlabel.\n");
        labeled_mini_image = mxGetData(bwlabel_return[0]); /* labeled image is a bwlabeled version of the mini image */
        temp = mxGetScalar(bwlabel_return[1]);
        nb_objects = (int)temp;
        
        /* if there is only 1 object in the image, continue */
        if(nb_objects > 1) {
            
            /* generate the mini edge image */
            mini_edge_image = calloc(mini_m*mini_n, sizeof(int));
            if(mini_edge_image == NULL)
                mexErrMsgTxt("Error: out of heap space\n");
            generate_sub_image(mini_edge_image, image_in, i_min, i_max, j_min, j_max, m);
            
            /* increment nb_objects to allow using it as lookup table */
            nb_objects++;
            /* calculate the object size of each labeled objects */
            object_size = calloc(nb_objects, sizeof(int));
            for(k = 0; k < mini_m*mini_n; k++) {
                if(labeled_mini_image[k] > 0) {
                    object_size[(int)labeled_mini_image[k]]++;
                }
            }
            
            /* find the max size object */
            winner_body = 0; max_val = 0;
            for(k = 1; k < nb_objects; k++) {
                if(object_size[k] > max_val) {
                    max_val = object_size[k];
                    winner_body = k;
                }
            }
            
            /*Create the matrix body_neighbors that holds all the numbers of the neighbors of each body */
            /* body_neighbors is a lookup table so row 0 and col 0 are unused */
            body_neighbors = calloc(highest_obj_nb*nb_objects, sizeof(int));
            /* loop over the edge pixels of image out mask */
            for(k = 0; k < mini_m*mini_n; k++) {
                if(mini_edge_image[k] == 0 || ((int)labeled_mini_image[k]) == winner_body) /* if this is not an edge pixel, or it is part of the winner body skip it */
                    continue;
                /* convert the linear index into (i,j), adjust i,j coords to refer to the full image */
                divresult = div (k,mini_m);
                j = (int)divresult.quot + j_min;
                i = (int)divresult.rem + i_min;
                if(i == 0 || j == 0 || i == (m-1) || j == (n-1)) /* if it is an edge pixel skip it */
                    continue;
                
                
                /* Check if the left neighbor pixel is not the background and is not object k in image_out */
                pixel = image_out[ij_to_lin(i,j-1,m)];
                if(pixel > 0 && pixel != obj_nb) {
                    body_neighbors[pixel + highest_obj_nb*((int)labeled_mini_image[k])]++;
                }
                
                /* Check if the top neighbor pixel is not the background and is not object k in image_out */
                pixel = image_out[ij_to_lin(i-1,j,m)];
                if(pixel > 0 && pixel != obj_nb) {
                    body_neighbors[pixel + highest_obj_nb*((int)labeled_mini_image[k])]++;
                }
                
                /* Check if the right neighbor pixel is not the background and is not object k in image_out */
                pixel = image_out[ij_to_lin(i,j+1,m)];
                if(pixel > 0 && pixel != obj_nb) {
                    body_neighbors[pixel + highest_obj_nb*((int)labeled_mini_image[k])]++;
                }
                
                /* Check if the bottom neighbor pixel is not the background and is not object k in image_out */
                pixel = image_out[ij_to_lin(i+1,j,m)];
                if(pixel > 0 && pixel != obj_nb) {
                    body_neighbors[pixel + highest_obj_nb*((int)labeled_mini_image[k])]++;
                }
            } /* end for loop */
            
            /* find the dominant neighbor of each labeled object in mask */
            neighbor_winner = calloc(nb_objects, sizeof(int));
            max_neighbors = calloc(nb_objects, sizeof(int));
            for(j = 1; j < nb_objects; j++) {
                for(i = 1; i < highest_obj_nb; i++) {
                    if(body_neighbors[ij_to_lin(i,j,highest_obj_nb)] > max_neighbors[j]) {
                        max_neighbors[j] = body_neighbors[ij_to_lin(i,j,highest_obj_nb)];
                        neighbor_winner[j] = i;
                    }
                }
            }
            
            /* loop over the image out and relabel the objects that need it */
            for(k = 0; k < mini_m*mini_n; k++) {
                /* if pixel(i,j) is a background pixel in labeled_image or belongs to the winner body: continue */
                if(((int)labeled_mini_image[k]) != 0 && ((int)labeled_mini_image[k]) != winner_body) {
                    /* convert the linear index into (i,j), adjust i,j coords to refer to the full image */
                    divresult = div (k,mini_m);
                    j = (int)divresult.quot + j_min;
                    i = (int)divresult.rem + i_min;
                    
                    /* if no neighbor is found for body with number labeled_image(i,j), delete the pixel and continue */
                    if(max_neighbors[((int)labeled_mini_image[k])] == 0) {
                        image_out[ij_to_lin(i,j,m)] = 0;
                    }else{
                        /* Otherwise renumber the body to the winner_neighbor */
                        image_out[ij_to_lin(i,j,m)] = neighbor_winner[((int)labeled_mini_image[k])];
                    }
                }
            }
            
            /* free matlab memory */
            free(body_neighbors);
            free(neighbor_winner);
            free(max_neighbors);
            free(mini_edge_image);
            
        } /* end if(nb_objects > 1) */
        
        mxDestroyArray(mask_array);
        mxDestroyArray(bwlabel_return[0]);
    } /* end for loop over the labeled objects in image_in */
    
    /* free image features memory */
    free(edge_image);
    free(cell_nbs);
    free(bounding_box);
    
    mxDestroyArray(connectivity);
}

void generate_sub_mask(mxLogical *mini_mask, const int *image_in, int i_min, int i_max, int j_min, int j_max, int m, int obj_nb) {
    register int k, i, j;
    int indx;
    
    /* copy over data from full mask */
    k = 0;
    for(j = j_min; j <= j_max; j++) {
        indx = j*m + i_max;
        for(i = (j*m + i_min); i <= indx; i++) {
            if(image_in[i] == obj_nb) {
                mini_mask[k] = 1;
            }
            k++;
        }
    }
}

void generate_sub_image(int *mini_mask, const int *image_in, int i_min, int i_max, int j_min, int j_max, int m) {
    register int k, i, j;
    int indx;
    
    /* copy over data from full mask */
    k = 0;
    for(j = j_min; j <= j_max; j++) {
        indx = j*m + i_max;
        for(i = (j*m + i_min); i <= indx; i++) {
            mini_mask[k++] = image_in[i];
        }
    }
}

void image_features(int *edgeImage, const int nb_cells, const int *cell_nbs, int *bounding_box, const int *imageIn, const int m, const int n) {
    
    register int k;
    int label, ii, i, j;
    int end_loc, start_loc, nb_elements;
    int *indexing;
    div_t divresult;
    
    /* cell_nbs are sorted list so (end-1) is the max value */
    /* allocate memory to create indexing vector (allows skipping labels) */
    indexing = calloc(cell_nbs[nb_cells-1], sizeof(int));
    /* populate the indexing vector */
    for(k = 0; k < nb_cells; k++) {
        indexing[cell_nbs[k]-1] = k+1;
    }
    
    nb_elements = (int)m*n;
    
    /* bounding_box(k,:) = [i_min i_max j_min j_max] */
    /* initialize bounding_box to: [m 0 n 0] */
    start_loc = 0;
    end_loc = nb_cells;
    for(k = start_loc; k < nb_cells; k++) {
        bounding_box[k] = m;
    }
    start_loc += nb_cells;
    end_loc += nb_cells;
    for(k = start_loc; k < end_loc; k++) {
        bounding_box[k] = 0;
    }
    start_loc += nb_cells;
    end_loc += nb_cells;
    for(k = start_loc; k < end_loc; k++) {
        bounding_box[k] = n;
    }
    start_loc += nb_cells;
    end_loc += nb_cells;
    for(k = start_loc; k < end_loc; k++) {
        bounding_box[k] = 0;
    }
    
    /* loop over the image */
    for(k = 0; k < nb_elements; k++) {
        /* if there is a non background pixel */
        if(imageIn[k] != 0) {
            /* record the label of the current pixel */
            ii = imageIn[k] - 1;
            label = indexing[ii];
            
            /* convert the image value to C index (start at 0) */
            label--;
            
            /* convert the linear index into (i,j) */
            divresult = div (k,m);
            j = (int)divresult.quot;
            i = (int)divresult.rem;
            
            /* update the bounding boxes */
            /* if i < i_min */
            ii = label;
            if(i < bounding_box[ii])
                bounding_box[ii] = i;
            /* if i > i_max */
            ii += nb_cells;
            if(i > bounding_box[ii])
                bounding_box[ii] = i;
            /* if j < j_min */
            ii += nb_cells;
            if(j < bounding_box[ii])
                bounding_box[ii] = j;
            /* if j > j_max */
            ii += nb_cells;
            if(j > bounding_box[ii])
                bounding_box[ii] = j;
            
            /* check for perimeter pixel */
            label = imageIn[k];
            if(i==0 || j==0 || i==(m-1) || j==(n-1) ||
                    imageIn[k-1]!=label || imageIn[k+1]!=label || imageIn[k-m]!=label || imageIn[k+m]!=label ||
                    imageIn[k-1-m]!=label || imageIn[k-1+m]!=label || imageIn[k+1-m]!=label || imageIn[k+1+m]!=label) {
                /* was a edge pixel */
                
                /* update edge image */
                edgeImage[k] = imageIn[k];
                
            }
        } /* end if(imageIn[k] != 0) */
    } /* end for(k = 0; k < nb_elements; k++) */
    
    /* free the dynamic memory */
    free(indexing);
}

int* unique(const int *list_in, int *nb_elements, int modulo) {
    
    register int k, j;
    int hash_val, found, nb_labels_found;
    int *list_out;
    Stack *array_stack, unique_stack;
    
    array_stack = calloc(modulo, sizeof(Stack));
    /* init the array of stacks to hold the unique elements */
    for(k = 0; k < modulo; k++) {
        array_stack[k] = Stack_Init(4);
    }
    
    /* Hash the input array values in order to unique them */
    for(k = 0; k < *nb_elements; k++) {
        if(list_in[k] == 0) /* If the list value is a zero, ignore it as background */
            continue;
        /* get the index in the hash table that the current value should be stored */
        hash_val = list_in[k]%modulo;
        found = 0;
        /* look to see if this value has already been added to the hash */
        for(j = 0; j < array_stack[hash_val].nb_el; j++) {
            if(array_stack[hash_val].data[j] == list_in[k]) {
                /* If the value was found, flip the flag and break out of the search loop */
                found = 1;
                break;
            }
        }
        if(found == 0) {
            /* If the value was not found in the hash table, add it */
            push(&array_stack[hash_val], list_in[k]);
        }
    }
    
    /* Pull the unique values from list_in out of the hash table */
    unique_stack = Stack_Init(100);
    for(k = 0; k < modulo; k++) {
        /* for each element in the hash table, copy out the values stored at that index */
        while(array_stack[k].nb_el > 0) {
            push(&unique_stack, pop(&array_stack[k]));
        }
        /* destroy the stack that was storing values at that location in the hash table */
        Destroy_Stack(array_stack[k]);
    }
    
    if(unique_stack.nb_el == 0) {
        /* if there were no nonzero elements in the array, free memory and return */
        Destroy_Stack(unique_stack);
        *nb_elements = 0;
        return NULL;
    }
    
    /* copy out the elements from the unique stack to the array */
    nb_labels_found = unique_stack.nb_el;
    /* allocate memory for the output unique list */
    list_out = calloc(nb_labels_found, sizeof(int));
    if(list_out == NULL)
        mexErrMsgTxt("Error: Out of heap space.\n");
    
    memcpy(list_out, unique_stack.data, nb_labels_found*sizeof(int));
    /* destroy the stack */
    Destroy_Stack(unique_stack);
    
    /* sort the output array using stdlib::quicksort */
    qsort(list_out, nb_labels_found, sizeof(int), compare);
    
    /* check that there are no negative labels in the input matrix */
    for(k = 0; k < nb_labels_found; k++) {
        if(list_out[k] < 0)
            mexErrMsgTxt("Invalid Label Number (nb < 0)");
    }
    
    *nb_elements = nb_labels_found;
    return list_out;
}

Stack Stack_Init(int new_size) {
    Stack S;
    S.size = new_size;
    S.data = calloc(S.size, sizeof(int));
    if(S.data == NULL)
        mexErrMsgTxt("Error: Out of heap space.");
    S.nb_el = 0;
    return S;
}

void push(Stack *S, int val) {
    if(S->nb_el >= S->size) {
        /* out of space in the stack, double the size */
        S->size = (int)2*S->size;
        S->data = realloc(S->data, sizeof(int)*S->size);
        if(S->data == NULL)
            mexErrMsgTxt("Error: Out of heap space.");
    }
    S->data[(S->nb_el)++] = val;
}

int pop(Stack *S) {
    if(S->nb_el <= 0)
        mexErrMsgTxt("Error: Attempt to pop empty stack.");
    /* update the nb of elements in the stack, pop the last one */
    return S->data[--(S->nb_el)];
}

void Destroy_Stack(Stack S) {
    free(S.data);
}

