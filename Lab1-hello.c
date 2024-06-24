#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (int argc, char* argv[])
{

  //TODO 3
  int thread_count = 6;

  //TODO 5 
  int myID;
  int num_threads;

  #pragma omp parallel num_threads (thread_count)//TODO 4 
    {

      //TODO 1
      myID = omp_get_thread_num(); 

      //TODO 2
      num_threads = omp_get_num_threads();
  
      printf("Hello from thread %d of %d\n", myID, num_threads);
    }

  return 0;
}


/*
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1
Hello from thread 0 of 1

TODO 1
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 3 of 1
Hello from thread 6 of 1
Hello from thread 5 of 1
Hello from thread 7 of 1
Hello from thread 4 of 1
Hello from thread 2 of 1
Hello from thread 0 of 1
Hello from thread 1 of 1

TODO 2
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 5 of 8
Hello from thread 7 of 8
Hello from thread 4 of 8
Hello from thread 6 of 8
Hello from thread 0 of 8
Hello from thread 2 of 8
Hello from thread 1 of 8
Hello from thread 3 of 8

TODO 4
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 0 of 6
Hello from thread 2 of 6
Hello from thread 4 of 6
Hello from thread 3 of 6
Hello from thread 5 of 6
Hello from thread 1 of 6

TODO 5
They become shared.
Since there is no sync mechanism, all the threads will write to it at the same time.
The OS will schedule the threads as it sees fit, thus it will produce differente outputs each execution. 

[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 5 of 6
Hello from thread 3 of 6
Hello from thread 3 of 6
Hello from thread 5 of 6
Hello from thread 5 of 6
Hello from thread 5 of 6

OR

[rm.barbosa@deeplearning01 OpenMP]$ ./lab1 
Hello from thread 3 of 6
Hello from thread 2 of 6
Hello from thread 2 of 6
Hello from thread 1 of 6
Hello from thread 2 of 6
Hello from thread 2 of 6
*/
