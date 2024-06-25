#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int computeX(int i)
{
  return i*1;
}

int main()
{
  int i;
  int N = 100;
  int my_sum=5, my_x, sum = 0;

#pragma omp parallel firstprivate(my_sum) private(my_x)//TODO1 //TODO3
  {

  #pragma omp for
    for (i=0; i< N; i++)  {
      my_x = computeX(i);
      my_sum += my_x;
    }

    //TODO 2
    #pragma omp critical
    sum+= my_sum;
   }
  printf("Sum is %d \n", sum);
  printf("Sum should be %d \n", (99*100)/2);

  return 0;
}


/*
TODO 1
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1_1 
Sum is 2356 
Sum should be 4950

TODO 2
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1_1 
Sum is 4950 
Sum should be 4950

TODO 3
my_sum = 5

[rm.barbosa@deeplearning01 OpenMP]$ ./lab1_1 
Sum is 4990 
Sum should be 4950 

TODO 4
[rm.barbosa@deeplearning01 OpenMP]$ ./lab1_1 
Sum is 4950 
Sum should be 4950
*/



//TODO 4
/*
int main()
{
 
  int N = 100;
  int i = 0 ; 
  int my_x, sum = 0;
   
#pragma omp parallel for reduction(+:sum) private(my_x)
  for (i=0; i< N; i++)  {
    my_x = computeX(i);
    sum += my_x;
  }
 
  printf("Sum is %d \n", sum);
  printf("Sum should be %d \n", (99*100)/2);

  return 0;
}
*/
