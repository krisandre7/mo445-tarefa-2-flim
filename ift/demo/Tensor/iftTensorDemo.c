//
// Created by Cesar Castelo on Apr 25, 2019
//

#include <ift.h>

int main(int argc, char** argv)
{
    if(argc != 7) {
        iftError("\nUsage: iftTensorDemo <...>\n"
                    "[1] input_tensor_a: Input tensor A (.csv)\n"
                    "[2] input_dim_array_a: Input tensor A dimension array (.csv)\n"
                    "[3] input_tensor_b: Input tensor B (.csv)\n"
                    "[4] input_dim_array_b: Input tensor B dimension array (.csv)\n"
                    "[5] output_tensor_c: Output tensor C (.csv)\n"
                    "[6] output_dim_array_c: Output tensor C dimension array (.csv)\n",
                 "iftTensorDemo.c");
    }

    /* read parameters */
    char *inputFilenameA = iftCopyString(argv[1]);
    char *inputDimArrayA = iftCopyString(argv[2]);
    char *inputFilenameB = iftCopyString(argv[3]);
    char *inputDimArrayB = iftCopyString(argv[4]);
    char *outputFilenameC = iftCopyString(argv[5]);
    char *outputDimArrayC = iftCopyString(argv[6]);

    /* read the tensors */
    printf("- Reading the input tensors ... "); fflush(stdout);
    iftTensor *A = iftReadTensorCsv(inputFilenameA, inputDimArrayA, ';');
    iftTensor *B = iftReadTensorCsv(inputFilenameB, inputDimArrayB, ';');
    printf("OK\n");

    printf("- Multiplying the tensors ... "); fflush(stdout);
    iftTensor *C = iftMultTensors(A, B);
    printf("OK\n");

    printf("- Writing the output tensors ... "); fflush(stdout);
    iftWriteTensorCsv(C, outputFilenameC, outputDimArrayC, ';');
    printf("OK\n");

    return 0;
}
