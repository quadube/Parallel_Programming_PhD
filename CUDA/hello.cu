#include <stdio.h>
/*
//=========== PART-1 ========================
__global__ void hello()
{

}

int main(void)
{

	hello<<< 1, 1 >>>();
	cudaDeviceSynchronize();

	printf("Hello World\n");
	return 0;
}
*/

/*
TODO 1 - Compile and run the code
[rm.barbosa@deeplearning01 2_hello]$ ./hello
Hello World

TODO 2 - comment first part, uncomment second
[rm.barbosa@deeplearning01 2_hello]$ ./hello
H
E
L
L
O
 
W
O
R
L
D
!

TODO 3 - change 12 to 16
[rm.barbosa@deeplearning01 2_hello]$ ./hello
H
E
L
L
O
 
W
O
R
L
D
!
H
E
L
L

TODO 4 - change 1,16 to 2,12
[rm.barbosa@deeplearning01 2_hello]$ ./hello
H
E
L
L
O
 
W
O
R
L
D
!
H
E
L
L
O
 
W
O
R
L
D
!

TODO 5 - change 2,12 to 2,16
[rm.barbosa@deeplearning01 2_hello]$ ./hello
H
E
L
L
O
 
W
O
R
L
D
!
H
E
L
L
H
E
L
L
O
 
W
O
R
L
D
!
H
E
L
L

*/



//=========== PART-2 ========================

__device__ const char *STR = "HELLO WORLD!";
const char STR_LENGTH = 12;

__global__ void hello()
{
	//every thread prints one character
	printf("%c\n", STR[threadIdx.x % STR_LENGTH]);
}

int main(void)
{

	hello<<< 2, 16>>>();
	cudaDeviceSynchronize();

	return 0;
}
