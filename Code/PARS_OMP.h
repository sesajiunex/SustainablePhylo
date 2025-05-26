//Header of the PARS class

#ifndef _PARS_H_
#define _PARS_H_
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <time.h>
#include "TreeInterface.h"
#include <fstream>
#include <omp.h>
#include <immintrin.h>
using namespace std;

#define REP_SIZE 2000 //Size of the input tree repository (in terms of phylogenetic trees). This value must be configured according to the number of trees in the repository file
#define MAX_SONS 10 //Maximum number of children in the topology. It can be configured according to the topological characteristics of the evaluated trees

//Optimized topological node structure. This is the basic type for the data structure that will contain the phylogenetic tree traversal for its processing by the kernel
typedef struct
{
        short number_of_sons; //Number of children of the node
        int sons_ids [MAX_SONS]; //Identifiers of the children node
}omp_node2;


class PARS
{
        TreeInterface** treePop; //Sets of phylogenetic trees to be evaluated
        int popSize; //number of trees
        int n_sequences; //number of input sequences
        int n_sites; //input sequence length
        string database; //database name
        SiteContainer* sites; //BIO++ dataset structure (with gaps)
        SiteContainer* completeSites; //BIO++ dataset structure (without gaps)
        Phylip* seqReader; //BIO++ sequence reader
        DNA* alphabet; //BIO++ DNA alphabet

        omp_node2* ocl_nodes; //Phylogenetic tree representation using our optimized topological node structure
        int ocl_n_nodes; //Number of nodes in the phylogeny
        int* internal_nodes; //Identifiers of internal nodes
        int num_internal_nodes; //Number of internal nodes

        char** fitch_sequences; //input dataset in hexadecimal codification
        char* line_fitch_sequences_simple; //input dataset in hexadecimal codification (char array)

        protected:
                double get_time(); //Get timestamp using omp_get_wtime
        public:
                /* Initialization procedures */
                PARS(string _database); //Constructor
                PhylogeneticTree* readTreeFromFile (FILE* file, int _repsize, int _id); //Reads the phylogenetic trees to be evaluated from file
                int genInitialPopPar (string fic_trees, int _popsize, int _repsize); //Initializes the phylogenetic tree objects (BIO++)
                TreeInterface** getTreePopulation(); //Returns the treePop array
                void initializeFitchSequences (); //Initialize the input sequences (hexadecimal code) in cl_char configuration
                void initializeParsTree (int _id); //Initialize the ocl_nodes structure (topology for kernel processing)
                void deleteAuxParsStructures(); //Deletes the ocl_nodes structure and initializes to 0 its relates variable
                int doOpenMPPARSModCPU (int _popsize, int _num_threads); //OpenMP code for CPU
                int runHost (string fic_trees, int _popsize, int _num_threads); //Main host method, interacts with the user and calls to the doOpenMPPARSMod functions
                ~PARS(); //Destructor
};

#endif  //_PARS_H_

