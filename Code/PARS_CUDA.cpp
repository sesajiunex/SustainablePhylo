#define MAX_NODES 1459

#define WARP_SIZE 32
#define MAX_SONS 4

#include "PARS_OMP.h"

__constant__ opencl_node2 tree[MAX_NODES]; 
__constant__ opencl_node2 tree2[MAX_NODES]; 


/**
 * CUDA Kernel Device code
 */

//PARAMETERS: tree to be evaluated, input dataset array in row-major order, sequence_length, number of internal nodes in the tree, local memory for parallel reduction, output parsimony score array
// REDUCE 6 : Any Block Size Reduction

__global__ void computeParsimonyReduce6Any (char4* fitch_sequences, const int num_sites, const int num_inner, int* parsimony_output)
{
    extern __shared__ int shared_parsimony_output[];
    int4 parc_parsimony = make_int4(0,0,0,0);
    int id =  blockIdx.x * blockDim.x + threadIdx.x ;
    char4 my_characters[MAX_NODES]; //the auxiliary character array is set to support the maximum number of sequences in the processed datasets, to reduce memory usage use the specific number of sequences in your dataset
    int i,j;
    char4 site_value;
    char4 son_value;
    char4 aux_value;
    short number_of_sons;
    int node_class;
    int node_id;
    char4 zero = make_char4(0,0,0,0);

	//Processing topology
    for (i=0; i<num_inner; i++)
    {
                number_of_sons = tree[i].number_of_sons;
                site_value = make_char4(31,31,31,31);
                for(j=0; j<number_of_sons; j++)
                {
                        node_class = tree[i].sons_ids[j] & 0x80000000;
                        node_id = tree[i].sons_ids[j] & 0x7FFFFFFF;
                        if(node_class==0x80000000)
                                son_value = my_characters[node_id];
                        else
                                son_value = fitch_sequences[num_sites*node_id+id];

                        aux_value = make_char4(site_value.x & son_value.x, site_value.y & son_value.y, site_value.z & son_value.z, site_value.w & son_value.w);
                        parc_parsimony.x = (aux_value.x == zero.x) ? (parc_parsimony.x+1) : parc_parsimony.x;
                        parc_parsimony.y = (aux_value.y == zero.y) ? (parc_parsimony.y+1) : parc_parsimony.y;
                        parc_parsimony.z = (aux_value.z == zero.z) ? (parc_parsimony.z+1) : parc_parsimony.z;
                        parc_parsimony.w = (aux_value.w == zero.w) ? (parc_parsimony.w+1) : parc_parsimony.w;
                        site_value.x = (aux_value.x == zero.x) ? (site_value.x | son_value.x) : aux_value.x;
                        site_value.y = (aux_value.y == zero.y) ? (site_value.y | son_value.y) : aux_value.y;
                        site_value.z = (aux_value.z == zero.z) ? (site_value.z | son_value.z) : aux_value.z;
                        site_value.w = (aux_value.w == zero.w) ? (site_value.w | son_value.w) : aux_value.w;
                }
                my_characters[i] = site_value;
    }

    //PERFORMING REDUCTION

    //each thread write his parcial parsimony in shared mem
    int tid=threadIdx.x;
    shared_parsimony_output[tid]=parc_parsimony.x+parc_parsimony.y+parc_parsimony.z+parc_parsimony.w;

    __syncthreads();

    // start the shared memory loop on the next power of 2 less
    // than the block size.  If block size is not a power of 2,
    // accumulate the intermediate sums in the remainder range.
    int floorPow2 = blockDim.x;

    if ( floorPow2 & (floorPow2-1) ) {
        while ( floorPow2 & (floorPow2-1) ) {
            floorPow2 &= floorPow2-1;
        }
        if ( tid >= floorPow2 ) {
            shared_parsimony_output[tid - floorPow2] += shared_parsimony_output[tid];
        }
        __syncthreads();
    }

    for ( int activeThreads = floorPow2>>1; activeThreads; activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            shared_parsimony_output[tid] += shared_parsimony_output[tid+activeThreads];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {
        parsimony_output[blockIdx.x] = shared_parsimony_output[0];
    }
}

__global__ void computeParsimonyReduce6Any2 (char4* fitch_sequences, const int num_sites, const int num_inner, int* parsimony_output)
{
    extern __shared__ int shared_parsimony_output[];
    int4 parc_parsimony = make_int4(0,0,0,0);
    int id =  blockIdx.x * blockDim.x + threadIdx.x ;
    char4 my_characters[MAX_NODES]; //the auxiliary character array is set to support the maximum number of sequences in the processed datasets, to reduce memory usage use the specific number of sequences in your dataset
    int i,j;
    char4 site_value;
    char4 son_value;
    char4 aux_value;
    short number_of_sons;
    int node_class;
    int node_id;
    char4 zero = make_char4(0,0,0,0);
 
    //Processing topology
    for (i=0; i<num_inner; i++)
    {
                number_of_sons = tree2[i].number_of_sons;
                site_value = make_char4(31,31,31,31);
                for(j=0; j<number_of_sons; j++)
                {
                        node_class = tree2[i].sons_ids[j] & 0x80000000;
                        node_id = tree2[i].sons_ids[j] & 0x7FFFFFFF;
                        if(node_class==0x80000000)
                                son_value = my_characters[node_id];
                        else
                                son_value = fitch_sequences[num_sites*node_id+id];
                        aux_value = make_char4(site_value.x & son_value.x, site_value.y & son_value.y, site_value.z & son_value.z, site_value.w & son_value.w);

                        parc_parsimony.x = (aux_value.x == zero.x) ? (parc_parsimony.x+1) : parc_parsimony.x;
                        parc_parsimony.y = (aux_value.y == zero.y) ? (parc_parsimony.y+1) : parc_parsimony.y;
                        parc_parsimony.z = (aux_value.z == zero.z) ? (parc_parsimony.z+1) : parc_parsimony.z;
                        parc_parsimony.w = (aux_value.w == zero.w) ? (parc_parsimony.w+1) : parc_parsimony.w;
                        site_value.x = (aux_value.x == zero.x) ? (site_value.x | son_value.x) : aux_value.x;
                        site_value.y = (aux_value.y == zero.y) ? (site_value.y | son_value.y) : aux_value.y;
                        site_value.z = (aux_value.z == zero.z) ? (site_value.z | son_value.z) : aux_value.z;
                        site_value.w = (aux_value.w == zero.w) ? (site_value.w | son_value.w) : aux_value.w;
                }
                my_characters[i] = site_value;
    }

    //PERFORMING REDUCTION

    //each thread write his parcial parsimony in shared mem
    int tid=threadIdx.x;
    shared_parsimony_output[tid]=parc_parsimony.x+parc_parsimony.y+parc_parsimony.z+parc_parsimony.w;

    __syncthreads();

    // start the shared memory loop on the next power of 2 less
    // than the block size.  If block size is not a power of 2,
    // accumulate the intermediate sums in the remainder range.
    int floorPow2 = blockDim.x;

    if ( floorPow2 & (floorPow2-1) ) {
        while ( floorPow2 & (floorPow2-1) ) {
            floorPow2 &= floorPow2-1;
        }
        if ( tid >= floorPow2 ) {
            shared_parsimony_output[tid - floorPow2] += shared_parsimony_output[tid];
        }
        __syncthreads();
    }

    for ( int activeThreads = floorPow2>>1; activeThreads; activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            shared_parsimony_output[tid] += shared_parsimony_output[tid+activeThreads];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {
        parsimony_output[blockIdx.x] = shared_parsimony_output[0];
    }
}


/**
 * Host main routine
 */

int main(int argc, char *argv[])
{
        if(argc != 4){
                printf("Input syntax error\n\t./PARS BD_fic trees_fic DEVICE\n");
                return 0;
        }

        printf("Starting...\n");
        string database=argv[1];
        string trees_fic=argv[2];
        int popsize = 2000; //Number of trees to evaluate, it must be configured according to the size of the tree repository
        int groupsize = 1024;
        PARS p(database);

        // Initialize the host inputs
        printf("Initializing input trees via BIO++...\n");
        p.genInitialPopPar(trees_fic, popsize, REP_SIZE); //REP_SIZE value must be changed according to the size of your trees repo.
        printf("Initializing input trees -- DONE\n");
        cudaDeviceProp prop;
        cudaSetDevice(atoi(argv[3])); //Change CUDA Device
        int dev;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        int CU_NUMBER = prop.multiProcessorCount;
        p.setCuNumber(CU_NUMBER);
        printf("SMs (Stream Multiprocessors): %d\n", CU_NUMBER); 
        printf("Initializing fitch sequences...\n");
        p.initializeFitchSequencesChar4();
        printf("Initializing fitch sequences -- DONE\n");
        int compressed_sites = (p.getFitchNSites()/4);

        if(p.getFitchNSites()%4!=0)
                compressed_sites++;
        if(compressed_sites%CU_NUMBER != 0)
        {
                compressed_sites = CU_NUMBER*(compressed_sites/CU_NUMBER) + CU_NUMBER;
        }
        size_t problem_size = compressed_sites;
        cout<<"PROBLEM SIZE: "<<problem_size<<endl;
        size_t work_group = groupsize;
        cout<<"WORK GROUP SIZE "<<work_group<<endl;
        if(problem_size % work_group != 0)
        {
                //Invalid work-group size detected, changing to a valid value
                work_group = problem_size/CU_NUMBER;
                if(work_group > 1024)
                {
                        int iter;
                        iter = problem_size/2;
                        while (problem_size%iter != 0 || iter > 1024)
                                iter--;
                        work_group=iter;
                }
                printf("Invalid workgroup size, setting value to %d\n", (int)work_group);
        }
        //Initializing the partial parsimony scores array (results of the reduction per workgroup)
        int output_size;
        if (compressed_sites%work_group==0)
                output_size = compressed_sites/work_group;
        else
                output_size = (compressed_sites/work_group)+(compressed_sites%work_group);

        int parsimonyVector [output_size];
        int accPars;
        //Obtaining allocation values for the buffers
        int max_ocl_n_nodes;
        int max_internal_nodes;
        int max_terminal_nodes;
        int aux_ocl_n_nodes;
        int aux_internal_nodes;
        int aux_terminal_nodes;
        TreeInterface** treePop = p.getTreePopulation();
        max_ocl_n_nodes = treePop[0]->getTree()->getInferredTree()->getNumberOfNodes();
        max_terminal_nodes = treePop[0]->getTree()->getInferredTree()->getNumberOfLeaves();
        max_internal_nodes = max_ocl_n_nodes - max_terminal_nodes;
        for (int i=1; i<popsize; i++)
        {
                aux_ocl_n_nodes = treePop[i]->getTree()->getInferredTree()->getNumberOfNodes();
                aux_terminal_nodes = treePop[i]->getTree()->getInferredTree()->getNumberOfLeaves();
                aux_internal_nodes = aux_ocl_n_nodes - aux_terminal_nodes;
                if (aux_ocl_n_nodes > max_ocl_n_nodes) max_ocl_n_nodes = aux_ocl_n_nodes;
                if (aux_terminal_nodes > max_terminal_nodes) max_terminal_nodes=aux_terminal_nodes;
                if (aux_internal_nodes > max_internal_nodes) max_internal_nodes=aux_internal_nodes;
        }

        //CUDA Streams Create
        cudaStream_t Stream1;
        cudaStream_t Stream2;
        cudaStreamCreate(&Stream1);
        cudaStreamCreate(&Stream2);

        // Allocate the device inputs/outputs
        printf("Allocate memory...\n");
        int fitch_nsequences = p.getFitchNSequences();
        char4* line_fitch_sequences_char4 = p.getLineFitchSequencesChar4();

        char4* d_FitchSequences = NULL;
        cudaMalloc((void **) &d_FitchSequences, sizeof(char4)*(fitch_nsequences*compressed_sites));

        int d_NumSites = compressed_sites;

        int* d_ParsimonyOutput = NULL;
        int* d_ParsimonyOutput2 = NULL;
        cudaMalloc((void **) &d_ParsimonyOutput, sizeof(int)*(output_size));
        cudaMalloc((void **) &d_ParsimonyOutput2, sizeof(int)*(output_size));
        printf("Allocate memory -- DONE\n");

        // Launch CUDA Kernel
        printf("Launch kernel...\n");
        int blocksPerGrid = problem_size / work_group;
        int threadsPerBlock = work_group;
        omp_node2* h_Tree;
        int d_NumInner;
        int ocl_n_nodes;
        omp_node2* h_Tree2;
    	int d_NumInner2;
    	int ocl_n_nodes2;

        for (int i=0; i<popsize; i=i+2)
        {
                /*****************************************************************************************************/
                /***************************** FIRST STREAM COMPUTE **************************************************/
                /*****************************************************************************************************/
                p.initializeParsTree(i, 0);
                h_Tree = p.getOclNodes(0);//ocl_nodes
                d_NumInner = p.getNumInternalNodes(0);//num_internal_nodes
                ocl_n_nodes = p.getOclNnodes(0);//ocl_n_nodes
                cudaMemcpyToSymbolAsync(tree, h_Tree, sizeof(omp_node2)*(ocl_n_nodes), 0, cudaMemcpyHostToDevice, Stream1);

                if(i==0)//First kernel call, dataset array is transfered now
                {
                        cudaMemcpyAsync(d_FitchSequences, line_fitch_sequences_char4, sizeof(char4)*(fitch_nsequences*compressed_sites), cudaMemcpyHostToDevice, Stream1);
                }

                //Kernel launch 1
                computeParsimonyReduce6Any<<<blocksPerGrid, threadsPerBlock, sizeof(int)*(2*work_group), Stream1>>>(d_FitchSequences, d_NumSites, d_NumInner, d_ParsimonyOutput);//Dinamic shared_parsimony

                if(i>0)//Processing partial parsimony scores - Second Stream
                {
                        // Copy the device result in device memory to the host result in host memory 2
                        cudaMemcpyAsync(parsimonyVector, d_ParsimonyOutput2, sizeof(int)*(output_size), cudaMemcpyDeviceToHost, Stream2);
                        cudaStreamSynchronize(Stream2);

                        //Adding partial parsimony scores 2
                        accPars=0;
                        for (int j=0; j<output_size; j++)
                                accPars = accPars + parsimonyVector[j];
                        cout<<"STREAM 2 PARSIMONY SCORE: "<<accPars<<endl;

                        p.deleteAuxParsStructures(1);
                }

                /*****************************************************************************************************/
                /***************************** SECOND STREAM COMPUTE *************************************************/
                /*****************************************************************************************************/
                p.initializeParsTree(i+1, 1);
                h_Tree2 = p.getOclNodes(1);//ocl_nodes
                d_NumInner2 = p.getNumInternalNodes(1);//num_internal_nodes
                ocl_n_nodes2 = p.getOclNnodes(1);//ocl_n_nodes
                cudaMemcpyToSymbolAsync(tree2, h_Tree2, sizeof(omp_node2)*(ocl_n_nodes2), 0, cudaMemcpyHostToDevice, Stream2);

                if(i==0)//First kernel call, dataset array is transfered now
                {
                        cudaMemcpyAsync(d_FitchSequences, line_fitch_sequences_char4, sizeof(char4)*(fitch_nsequences*compressed_sites), cudaMemcpyHostToDevice, Stream2);
                }

                //Kernel launch 2
                computeParsimonyReduce6Any2<<<blocksPerGrid, threadsPerBlock, sizeof(int)*(2*work_group), Stream2>>>(d_FitchSequences, d_NumSites, d_NumInner2, d_ParsimonyOutput2);//Dinamic shared_parsimony

                //Processing partial parsimony scores - First Stream

                // Copy the device result in device memory to the host result in host memory 1
                cudaMemcpyAsync(parsimonyVector, d_ParsimonyOutput, sizeof(int)*(output_size), cudaMemcpyDeviceToHost, Stream1);
                cudaStreamSynchronize(Stream1);

                //Adding partial parsimony scores 1
                accPars=0;
                for (int j=0; j<output_size; j++)
                        accPars = accPars + parsimonyVector[j];
                cout<<"STREAM 1 PARSIMONY SCORE: "<<accPars<<endl;

                p.deleteAuxParsStructures(0);

        }
        //Processing partial parsimony scores - Second Stream (Last iteration)

        // Copy the device result in device memory to the host result in host memory 2
        cudaMemcpyAsync(parsimonyVector, d_ParsimonyOutput2, sizeof(int)*(output_size), cudaMemcpyDeviceToHost, Stream2);
        cudaStreamSynchronize(Stream2);

        //Adding partial parsimony scores 2
        accPars=0;
        for (int j=0; j<output_size; j++)
                accPars = accPars + parsimonyVector[j];
        cout<<"STREAM 2 PARSIMONY SCORE: "<<accPars<<endl;

        p.deleteAuxParsStructures(1);

        printf("Launch kernel -- DONE\n");

        // Free device global memory
        cudaFree(d_FitchSequences);
        cudaFree(d_ParsimonyOutput);
        cudaFree(d_ParsimonyOutput2);

        ////CUDA Streams Destroy
        cudaStreamDestroy(Stream1);
        cudaStreamDestroy(Stream2);

        printf("Finish!!!\n");

        return 0;
}
