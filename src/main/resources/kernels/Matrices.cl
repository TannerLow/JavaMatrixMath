__kernel void elementWiseMultiply(__global const float *a,
                                  __global const float *b,
                                  __global float *c,
                                  const int rowSize)
{
    int globalRow = get_global_id(0);
    int index = globalRow * rowSize;
    for (int col = 0; col < rowSize; col++) {
        c[index] = a[index] * b[index];
        index++;
    }
}

__kernel void elementWiseMultiplySlow(__global const float *a,
                                      __global const float *b,
                                      __global float *c,
                                      const int rowSize)
{
    int gid = get_global_id(0);
    c[gid] = a[gid] * b[gid];
}

// Matrix multiplication: C = A * B.
__kernel void
matrixMultiply(__global float* C,
               __global float* A,
               __global float* B,
               const int sharedDimension,
               const int bCols)
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);

    // value stores the element that is
    // computed by the thread
    float value = 0;
    for (int k = 0; k < sharedDimension; ++k)
    {
        float elementA = A[globalRow * sharedDimension + k];
        float elementB = B[k * bCols + globalCol];
        value += elementA * elementB;
    }

    // Write the matrix to device memory each
    // thread writes one element
    C[globalRow * bCols + globalCol] = value;
}

// Add row to rows: C = A[row] + B, for all rows.
__kernel void
addRowToRows(__global float* C,
             __global float* A,
             __global float* B,
             const int rowSize)
{
    int globalRow = get_global_id(0);

    int offset = globalRow * rowSize;

    // one thread per row
    for (int i = 0; i < rowSize; i++)
    {
        C[offset + i] = A[offset + i] + B[i];
    }
}

// Add col to cols: C = A[i][j] + B, for all cols j.
__kernel void
addColToCols(__global float* C,
             __global float* A,
             __global float* B,
             const int rowSize)
{
    int globalRow = get_global_id(0);

    int offset = globalRow * rowSize;

    // one thread per row of A
    for (int i = 0; i < rowSize; i++)
    {
        C[offset + i] = A[offset + i] + B[globalRow];
    }
}

// Relu: output = ReLu(A).
__kernel void
relu(__global float* output,
     __global float* input,
     const int rowSize)
{
    int globalRow = get_global_id(0);

    // one thread per row
    for (int i = 0; i < rowSize; i++)
    {
        int index = globalRow * rowSize + i;

        // output[i] = max(output[i], 0)
        float value = input[index];
        float newValue = 0;
        if(value > 0) {
            newValue = value;
        }

        output[index] = newValue;
    }
}

// Add row to rows: C = exp(A[i]) for all rows i.
__kernel void softmax(__global float* output,
                      __global float* input,
                      const int rowSize)
{
    int globalRow = get_global_id(0);

    int offset = globalRow * rowSize;

    // get the max value of the row
    float max = -3.4028235E37f;
    float value;
    for (int i = 0; i < rowSize; i++) {
        value = input[offset + i];
        if(value > max) {
            max = value;
        }
    }

    // Calculate sum of exponentials of input elements
    float sum = 0.0f;
    for (int i = 0; i < rowSize; i++) {
        sum += exp(input[offset + i] - max);
    }

    // Calculate softmax for each element
    int index;
    for (int i = 0; i < rowSize; i++) {
        index = offset + i;
        output[index] = exp(input[index] - max) / sum;
    }
}
