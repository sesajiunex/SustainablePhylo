#include "PARS_OMP.h"


/**
*       Constructor, initializes variables and data structures. Parameters:
*       _database: string containing the name of the input dataset
*/
PARS::PARS(string _database)
{
    treePop = NULL;
    popSize=0;
    n_sequences=0;
    n_sites=0;
    database = _database;

    alphabet=NULL;
    seqReader=NULL;
    sites=NULL;
    completeSites=NULL;

    fitch_sequences=NULL;
    ocl_nodes=NULL;
}

/**
*       Method that returns a timestamp using time.h
*/
double PARS::get_time (void)
{
                return omp_get_wtime();
}


/**
*       Method that reads from a file the Newick code of a tree to be evaluated. Parameters:
*       file: input tree file / repository
*       _repsize: size of the input tree file / repository (in terms of number of phylogenetic trees contained)
*       _id: position of the tree to be read
*/
PhylogeneticTree* PARS::readTreeFromFile (FILE* file, int _repsize, int _id)
{
        char character;
        int numtrees=0;
        NewickCode auxCode;
        //We suppose the file has been opened
        if (file == NULL) return NULL;
        if (_id < 0 || _id >= _repsize) return NULL;
        fseek(file, 0L, SEEK_SET);
        character = fgetc(file);
        //Reading until we find the tree
        while (character != EOF && numtrees != _id)
        {
                if (character == ';')
                {
                        numtrees++;
                }
                character = fgetc(file);
        }
        if (numtrees == _id)
        {
                if (_id == 0) fseek(file, 0L, SEEK_SET);
                //Reading the tree
                if (!auxCode.readNewick (file)) return NULL;
                PhylogeneticTree* tree= new PhylogeneticTree (_id, auxCode);
                return tree;
        }
        else return NULL;

}

/**
*       Method that reads initializes the phylogenetic trees to be evaluated (TreeInterface objects). Parameters:
*       fic_trees: name of the input tree file / repository
*       _popsize: number of trees to be considered
*       _repsize: size of the input tree file / repository (in terms of number of phylogenetic trees contained)
*/
int PARS::genInitialPopPar (string fic_trees, int _popsize, int _repsize)
{
    int i,j;
    int sel;
    FILE* file;
    int selected_trees [_popsize];
    PhylogeneticTree** trees;
    bool already_sel;
    //Reading tree file
    file = fopen (fic_trees.c_str(), "rt");
    if (file == NULL) return 0;
    //Initializing tree population
    popSize = _popsize;
    treePop = new TreeInterface* [_popsize];
    //Getting the popSize initial trees
    i=0;
    trees = new PhylogeneticTree* [_popsize];
    //Reading trees
    while (i<_popsize)
    {
                        sel=i%_repsize;
                        trees[i] = readTreeFromFile (file, _repsize, sel);

                        if (trees[i]==NULL) return 0;
                        selected_trees[i]=trees[i]->getId();
                        i++;
    }
    //Closing file
    fclose(file);
    //Initializing BIO++ objetcs
    alphabet = new DNA ();
    seqReader= new Phylip (false, false);
    sites=seqReader->readAlignment(database, alphabet);
    //Initializing TreeInterface objetcs
    for (i=0;i<_popsize;i++)
    {
                        trees[i]->initialize(sites, completeSites);
                        trees[i]->initializeTree();
                        treePop[i]=new TreeInterface (i,*trees[i]);
                        //trees[i]->setScores();
    }
    //Finishing
    for (i=0;i<_popsize;i++)
    {
                if (trees[i] != NULL) delete (trees[i]);
    }
    delete(trees);
    return 1;
}

/**
*       Method that returns the input trees set.
*/
TreeInterface** PARS::getTreePopulation()
{
        return treePop;
}

/**
*       Method that initializes the input sequences (hexadecimal code) in char configuration
*/
void PARS::initializeFitchSequences ()
{
        int i,j;
        //Getting input dataset sizes
        n_sequences = sites->getNumberOfSequences ();
        n_sites = sites->getNumberOfSites();
        //Initializing dataset variable
        fitch_sequences = new char* [n_sequences];
        for (i=0; i<n_sequences; i++)
        {
                fitch_sequences[i]=new char [n_sites];
        }
        //Converting the input dataset (BIO++ implementation) to our codification (fitch_sequences)
        string aux_sequence;
        for (i=0; i<n_sequences; i++)
        {
                aux_sequence = sites->toString(i);

                for(j=0; j<n_sites; j++)
                {
                        switch(aux_sequence[j])
                        {
                                case '-': fitch_sequences[i][j]=0x10; break;
                                case 'A': fitch_sequences[i][j]=0x08; break;
                                case 'C': fitch_sequences[i][j]=0x04; break;
                                case 'G': fitch_sequences[i][j]=0x02; break;
                                case 'T': fitch_sequences[i][j]=0x01; break;
                                case 'M': fitch_sequences[i][j]=0xC; break;
                                case 'R': fitch_sequences[i][j]=0xA; break;
                                case 'W': fitch_sequences[i][j]=0x09; break;
                                case 'S': fitch_sequences[i][j]=0x06; break;
                                case 'Y': fitch_sequences[i][j]=0x05; break;
                                case 'K': fitch_sequences[i][j]=0x03; break;
                                case 'V': fitch_sequences[i][j]=0xE; break;
                                case 'H': fitch_sequences[i][j]=0xD; break;
                                case 'D': fitch_sequences[i][j]=0xB; break;
                                case 'B': fitch_sequences[i][j]=0x07; break;

                                default: fitch_sequences[i][j]=0xF; break;

                        }
                }
        }
        //line_fitch_sequences: char container of the dataset in row-major order
        line_fitch_sequences_simple = new char[n_sequences*n_sites];
                int k;
        k=0;
        for (i=0; i<n_sequences; i++)
        {
            for(j=0; j<n_sites; j++)
            {
                    line_fitch_sequences_simple[k]=fitch_sequences[i][j];
                    k++;
            }
        }
}

/**
*       Method that initializes the ocl_nodes structure (topology for kernel processing). Post-order tree traversal. Parameters:
*       _id: identifier of the tree to be initialized
*/
void PARS::initializeParsTree (int _id)
{
        int i,j;
        int current_inner;
        int aux_nsons;
        int aux_son_id;
        string leaf_name;
        TreeTemplate<Node>*treeData;
        vector<Node *> nodes;

        //Obtaining information about the phylogenetic topology
        treeData = treePop[_id]->getTree()->getInferredTree();
        nodes = treeData->getNodes();
        num_internal_nodes = treeData->getInnerNodesId().size();
        int inner_index [nodes.size()];
        ocl_nodes = new omp_node2 [num_internal_nodes];
        ocl_n_nodes=num_internal_nodes;

        current_inner = 0;
        int* aux_term_pos = treePop[_id]->getTree()->getTerminalPositions();
        //TREE TRAVERSAL
        for(i=0; i<nodes.size(); i++)
        {
                aux_nsons = nodes[i]->getNumberOfSons();
                if (aux_nsons!=0)
                {
                        //INNER NODES
                        ocl_nodes[current_inner].number_of_sons = aux_nsons;
                        inner_index[i]=current_inner;
                        for (j=0; j<aux_nsons; j++)
                        {
                                aux_son_id = nodes[i]->getSonsId()[j];
                                if(nodes[aux_son_id]->getNumberOfSons()==0)
                                {
                                        //TERMINAL CHILD
                                        ocl_nodes[current_inner].sons_ids[j]=aux_term_pos[aux_son_id];
                                        ocl_nodes[current_inner].sons_ids[j]=ocl_nodes[current_inner].sons_ids[j]|(0<<31);
                                }
                                else
                                {
                                        //INNER CHILD
                                        ocl_nodes[current_inner].sons_ids[j]=inner_index[aux_son_id];
                                        ocl_nodes[current_inner].sons_ids[j]=ocl_nodes[current_inner].sons_ids[j]|(1<<31);
                                }
                        }
                        current_inner++;
                }
        }
}

/**
*       Method that deletes the ocl_nodes structure and initializes to 0 its relates variable.
*/
void PARS::deleteAuxParsStructures()
{
        delete(ocl_nodes);
        num_internal_nodes=0;
        ocl_n_nodes=0;
}

/**
*       Method that runs the OpenMP parsimony code for CPU in char configuration. Parameters:
*       _popsize: number of trees to be processed
*       _num_threads: number of OpenMP threads
*/

int PARS::doOpenMPPARSModCPU (int _popsize, int _num_threads)
{
        int i,u,m,w,j,k,l;

        initializeFitchSequences();

        //COLUMN-MAJOR ORDER VARIABLE FOR THE CPU DATASET
        char **line_fitch_sequences_simple2;
        const int v = 32;
        int n_sites_vectorizado= std::ceil((double)(n_sites)/v);
        int length_lfso = n_sites_vectorizado*n_sequences;

        char line_fitch_sequences_omp[length_lfso][v] __attribute__((aligned (32)));
        //Allocating memory for the vectorized dataset (sequences)
        line_fitch_sequences_simple2 = new char* [length_lfso];
        for (i=0; i<length_lfso; i++)
        {
                line_fitch_sequences_simple2[i] = new char [v];
        }

        //Vectorizing the dataset (sequences)
        w=0;
        for (i=0; i < n_sequences; i++)
        {
                for(u = 0; u < n_sites; u+=v)
                {
                        for(m = 0; m < v; m++)
                        {
                                if(u+m < n_sites)
                                        line_fitch_sequences_simple2[w][m] = fitch_sequences[i][u+m];
                                else
                                        line_fitch_sequences_simple2[w][m] = 0x10;
                        }
                        w++;
                }
        }
        //ORGANIZING COLUMN-MAJOR ORDER FOR THE CPU DATASET
        w=0;
        for (i=0; i<n_sites_vectorizado; i++){
	        for (u=0; u<n_sequences; u++){
	            for (m=0; m<v; m++){
	                        line_fitch_sequences_omp[w][m] = line_fitch_sequences_simple2[u*n_sites_vectorizado+i][m];
	                }
	            w++;
	        }
        }
        double parsimonyVector [_num_threads];
        for (i=0; i<_num_threads; i++)
                parsimonyVector[i] = 0;

        #pragma omp parallel num_threads(_num_threads) default (none) shared (_popsize, _num_threads, parsimonyVector, line_fitch_sequences_omp, n_sites_vectorizado) private(i, j, k, l)
        {
        //Evaluation loop
        for (i=0; i<_popsize; i++)
        {
                //Initializing omp_node topology
                #pragma omp single
                {
                        initializeParsTree(i);
                }
                int Parsimony = 0;

                char my_characters[num_internal_nodes][v] __attribute__((aligned (32)));
                int indice;
                int accPars=0;
                int partParsimony[8];
                for (j=0; j<8; j++)
                        partParsimony[j]=0;
                char* aux_vec = new char[v];
                int num_sons; //number of children of the node currently processed by the thread
                int node_class; //type of node (leaf or internal)
                int node_id; //identifier of the node currently processed by the thread
                __m256i and_site_son; //auxiliar variable for fitch operations
                __m256i or_site_son; //auxiliar variable for fitch operations
                __m256i site_value; //variable to store the state calculated for a node
                __m256i son_value; //variable to store the state read from a child node
                __m256i if_zero; //variable to store the if condition
                __m256i if_and;
                __m256i if_or;
                __m256i pars_mask = _mm256_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF);
                __m256i aux_pars; //variable to store the pars increment
                __m256i pars = _mm256_setzero_si256(); //variable to store the pars
                __m256i zeros = _mm256_setzero_si256(); //variable 0s
                __m256i ones = _mm256_set1_epi8(0x01); //variable 1s

                #pragma omp for schedule (guided)
                for (j = 0; j < n_sites_vectorizado; j++)
                {
                                for (k=0; k<num_internal_nodes; k++)
                                {
                                        site_value = _mm256_set1_epi8(0xFF);
                                        num_sons = ocl_nodes[k].number_of_sons;
                                        for (l=0; l<num_sons; l++)
                                        {
                                                node_class = ocl_nodes[k].sons_ids[l] & 0x80000000;
                                                node_id = ocl_nodes[k].sons_ids[l] & 0x7FFFFFFF;
                                                indice = n_sequences*j+node_id;

                                                if (node_class == 0x80000000){
                                                        //LOAD
                                                        son_value = _mm256_load_si256((const __m256i*)my_characters[node_id]);
                                                }
                                                else{
                                                        //LOAD
                                                        son_value = _mm256_load_si256((const __m256i*)line_fitch_sequences_omp[indice]);
                                                }

                                                //Compute and / or between site_value and son _value
                                                and_site_son = _mm256_and_si256(site_value, son_value);
                                                or_site_son = _mm256_or_si256(site_value,son_value);

                                                //Compute masks for if ((site_value & son_value) == 0) cases
                                                if_zero = _mm256_cmpeq_epi8(and_site_son,zeros);  

                                                //Identify cases that imply P(T)+++
                                                aux_pars = _mm256_and_si256(if_zero,ones);

                                                //Adding cases
                                                aux_pars =_mm256_sad_epu8(zeros,aux_pars);

                                                // Horizontal addition
                                                aux_pars = _mm256_hadd_epi16(aux_pars, _mm256_permute2x128_si256(aux_pars, aux_pars, 1));
                                                aux_pars = _mm256_hadd_epi16(aux_pars, aux_pars);
                                                aux_pars = _mm256_hadd_epi16(aux_pars, aux_pars);
                                                aux_pars = _mm256_hadd_epi16(aux_pars, aux_pars);
                                                Parsimony = Parsimony + _mm256_extract_epi16 (aux_pars, 0);

												//Computing locations of or states
                                                if_or = _mm256_and_si256(if_zero,or_site_son);

                                                //Combine and / or states
                                                site_value=_mm256_or_si256(if_or,and_site_son);
                                        }

                                        //STORE
                                        _mm256_store_si256((__m256i*)my_characters[k],site_value);
                                }
                }
                parsimonyVector[omp_get_thread_num()]=Parsimony;
                #pragma omp barrier
                #pragma omp single
                {

                        accPars=0;
                        for (j=0; j<_num_threads; j++)
                                accPars = accPars + parsimonyVector[j];
                        printf("PARSIMONY SCORE %d\n", accPars);

                        deleteAuxParsStructures();
                }
        }
        }
        delete(line_fitch_sequences_simple2);
        return 0;
}

/**
*       Main host method, interacts with the user and calls to the doOpenMPPARSModCPU functions. Parameters:
*       fic_trees: name of the input file containing the phylogenies to be evaluated
*       _popsize: number of trees to be processed
*       _num_threads: number of threads
*/

int PARS::runHost (string fic_trees, int _popsize, int _num_threads)
{
          int i;
          char v;
          char Buffer[1024];
          //Reading and initializing input trees
          printf("Initializing input trees via BIO++\n");
          genInitialPopPar(fic_trees, _popsize, REP_SIZE);
          printf("Initializing input trees -- DONE\n");

          printf("STARTING ON CPU DEVICE\n");
          doOpenMPPARSModCPU (_popsize, _num_threads);

                if(fitch_sequences!=NULL)
                {
                        for (i=0; i<n_sequences; i++)
                                        delete fitch_sequences[i];
                        delete (fitch_sequences);
                }
                if (treePop != NULL)
                {
                          for (i=0; i<popSize; i++)
                          {
                                  if (treePop[i]!=NULL) delete (treePop[i]);
                          }
                          delete(treePop);
                }
                return 0;
}

/** Destructor */
PARS::~PARS()
{


}

