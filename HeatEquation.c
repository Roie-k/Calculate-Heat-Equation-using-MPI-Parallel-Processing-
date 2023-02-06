#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define Columns      10
#define Total_Rows  6        // "global" row count
#define NoProc            4        // number of processors
#define Rows (Total_Rows/NoProc)  // number of real local Rows
// communication tags
#define DOWN     0
#define UP       1  
#define tolerance 0.01

double NewMat[Rows+2][Columns+2];
double A[Rows+2][Columns+2];
void initialize(int numberPE, int my_ID);
void track_progress(int iter);
int main(int argc, char *argv[]) {
	void prtdat();
    int i, j;
    int MAX_iter;
    int iteration=1;
    double dA;
    struct timeval start_time, stop_time, elapsed_time;
    int        numberPE;                // number of Processors
    int        my_ID;           // my Processor number
    double     dA_global=10;       // delta t across all Processors
    MPI_Status status;              // status returned by MPI calls
    // MPI startup routines
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
    MPI_Comm_size(MPI_COMM_WORLD, &numberPE);
    // verify only NoProc Processors are being used
    if(numberPE != NoProc) {
      if(my_ID==0) {
        printf("This code must be run with %d Processors\n", NoProc);
      }
      MPI_Finalize();
      exit(1);
    }
    // PE 0 asks for input
    if(my_ID==0) {
      printf("Maximum iterations [100-4000]?\n");
      fflush(stdout); 
      scanf("%d", &MAX_iter);
    }
    // bcast max iterations to other Processors
    MPI_Bcast(&MAX_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
    if (my_ID==0) gettimeofday(&start_time,NULL);

    initialize(numberPE, my_ID);
	
	FILE *fp;
	fp = fopen("INPUT.text", "w");
	for (i = 0; i < Total_Rows; i++) {
 	 for (j = 0; j < Columns; j++) {
   	 fprintf(fp, "%.2lf ", A[i][j]);
  	  }
	fprintf(fp, "\n");
  	}
	fclose(fp);

    while ( dA_global > tolerance && iteration <= MAX_iter ) {
        // main calculation: average my four neighbors
        for(i = 0; i < Total_Rows; i++) {
            for(j = 0; j < Columns; j++) {
                NewMat[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] +
                                            A[i][j+1] + A[i][j-1]);
            }
        }		
        // COMMUNICATION PHASE: send ghost Rows for next iteration
        // send bottom real row down
        if(my_ID != numberPE-1){             //unless we are bottom Processor
            MPI_Send(&NewMat[Rows][1], Columns, MPI_DOUBLE, my_ID+1, DOWN, MPI_COMM_WORLD);
        }
        // receive the bottom row from above into our top ghost row
        if(my_ID != 0){                  //unless we are top Processor
            MPI_Recv(&A[0][1], Columns, MPI_DOUBLE, my_ID-1, DOWN, MPI_COMM_WORLD, &status);
        }
        // send top real row up
        if(my_ID != 0){                    //unless we are top Processor
            MPI_Send(&NewMat[1][1], Columns, MPI_DOUBLE, my_ID-1, UP, MPI_COMM_WORLD);
        }
        // receive the top row from below into our bottom ghost row
        if(my_ID != numberPE-1){             //unless we are bottom Processor
            MPI_Recv(&A[Rows+1][1], Columns, MPI_DOUBLE, my_ID+1, UP, MPI_COMM_WORLD, &status);
        }
        dA = 0.0;

        for(i = 1; i <= Rows; i++){
            for(j = 1; j <= Columns; j++){
	        dA = fmax( fabs(NewMat[i][j]-A[i][j]), dA);
	        A[i][j] = NewMat[i][j];
            }
        }
        // find global dA                                                        
        MPI_Reduce(&dA, &dA_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dA_global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       // periodically print test values - only for Processor in lower corner
        if((iteration % 20) == 0) {
            if (my_ID == numberPE-1){
                track_progress(iteration);
	    }
        }
	iteration++;
    }	
    // Slightly more accurate timing and cleaner output 
    MPI_Barrier(MPI_COMM_WORLD);
	
    // Processor 0 finish timing and output values
    if (my_ID==0){
        gettimeofday(&stop_time,NULL);
	timersub(&stop_time, &start_time, &elapsed_time);

	FILE *fptr;
	fptr = fopen("OUTPUT.text", "w");
	for (i = 0; i < Total_Rows; i++) {
 	 for (j = 0; j < Columns; j++) {
   	 fprintf(fptr, "%.2lf ", NewMat[i][j]);
  	  }
	fprintf(fptr, "\n");
  	}
	fclose(fptr);

	printf("\nMax error at iteration %d was %f\n", iteration-1, dA_global);
	printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
}
    MPI_Finalize();
}
void initialize(int numberPE, int my_ID){
    double tMin, tMax;  //Local boundary limits
    int i,j;
	for(i = 0; i <= Total_Rows; i++){
        for (j = 0; j <= Columns; j++){
            A[i][j] = 30.0;
        }
    }
	for(i = 0; i <= Total_Rows; i++){
        for (j = 0; j <= Columns; j++){
            A[i][0] = 60.0;
		A[0][j] = 60.0;
		A[i][Columns-1] = 60.0;
		A[Total_Rows-1][j] = 60.0;
        }
    }
	// Set Middle Temperature to 20.0
	for(i = Total_Rows/2-1; i <= Total_Rows/2; i++){ 
        for (j = Columns/2-1; j <= Columns/2; j++){
            A[i][j] = 20.0;
        }
    }
	for(i = 0; i <= 1; i++){ 
        for (j = 0; j <= 2; j++){
            A[i][j] = 60.0;
        }
    }
	for(i = 4; i <= 5; i++){ 
        for (j = 0; j <= 2; j++){
            A[i][j] = 60.0;
        }
    }
	for(i = 4; i <= 5; i++){ 
        for (j = 7; j <= 9; j++){
            A[i][j] = 60.0;
        }
    }
	for(i = 0; i <= 1; i++){ 
        for (j = 7; j <= 9; j++){
            A[i][j] = 60.0;
        }
    }	
    // Local boundry condition endpoints
    tMin = (my_ID)*100.0/numberPE;
    tMax = (my_ID+1)*100.0/numberPE;
	}
// only called by last Processor
void track_progress(int iteration) {
    int i;
    printf("Iteration number: %d \n", iteration);
}

