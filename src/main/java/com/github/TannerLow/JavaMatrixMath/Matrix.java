package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.JavaMatrixMath.Exceptions.DimensionsMismatchException;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clSetKernelArg;

public class Matrix {
    public final int rows;
    public final int cols;
    public final float[] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }

    public Matrix(int rows, int cols, float[] data) {
        this.rows = rows;
        this.cols = cols;
        if(data.length == rows * cols) {
            this.data = data;
        }
        else {
            this.data = new float[rows * cols];
        }
    }

    public Matrix multiply(Matrix other) throws DimensionsMismatchException {
        if(cols != other.rows) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {other.rows, other.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        Matrix result = new Matrix(rows, other.cols);

        for(int row = 0; row < this.rows; row++) {
            for(int otherCol = 0; otherCol < other.cols; otherCol++) {
                float sum = 0;
                for(int col = 0; col < this.cols; col++) {
                    sum += data[row * this.cols + col] * other.data[col * other.cols + otherCol];
                }
                result.data[row * result.cols + otherCol] = sum;
            }
        }

        return result;
    }

    public Matrix addRowToRows(Matrix row) throws DimensionsMismatchException {
        if(cols != row.cols) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {row.rows, row.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        Matrix result = new Matrix(rows, cols);

        for(int currentRow = 0; currentRow < rows; currentRow++) {
            for(int col = 0; col < cols; col++) {
                int index = currentRow * cols + col;
                result.data[index] = data[index] + row.data[col];
            }
        }

        return result;
    }

    public Matrix addColToCols(Matrix col) throws DimensionsMismatchException {
        if(rows != col.rows) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {col.rows, col.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        Matrix result = new Matrix(rows, cols);

        for(int row = 0; row < rows; row++) {
            for(int currentCol = 0; currentCol < cols; currentCol++) {
                int index = row * cols + currentCol;
                result.data[index] = data[index] + col.data[row];
            }
        }

        return result;
    }

    public Matrix relu() {
        Matrix result = new Matrix(rows, cols);

        for(int i = 0; i < data.length; i++) {
            result.data[i] = Math.max(data[i], 0);
        }

        return result;
    }

    public Matrix vectorizedReluDerivative() {
        Matrix result = new Matrix(rows, cols);

        for(int i = 0; i < data.length; i++) {
            if(data[i] > 0) {
                result.data[i] = 1;
            }
        }

        return result;
    }

    public Matrix horizontalSoftmax() {
        Matrix result = new Matrix(rows, cols);

        float[] buffer = new float[rows];
        for(int row = 0; row < rows; row++) {
            int offset = row * cols;

            // calculate the max values
            buffer[row] = -Float.MAX_VALUE;
            for(int i = 0; i < cols; i++) {
                float value = data[offset + i];
                if(value > buffer[row]) {
                    buffer[row] = value;
                }
            }

            // calculate the sums
            float sum = 0;
            float max = buffer[row];
            for(int i = 0; i < cols; i++) {
                sum += Math.exp(data[offset + i] - max);
            }

            // calculate the softmax vectors
            for(int i = 0; i < cols; i++) {
                result.data[offset + i] = (float) (Math.exp(data[offset + i] - max) / sum);
            }
        }

        return result;
    }

    public Matrix verticalSoftmax() {
        Matrix result = new Matrix(rows, cols);

        float[] buffer = new float[cols];
        for(int col = 0; col < cols; col++) {
            // calculate the max values
            buffer[col] = -Float.MAX_VALUE;
            for(int i = 0; i < rows; i++) {
                float value = data[col + i * cols];
                if(value > buffer[col]) {
                    buffer[col] = value;
                }
            }

            // calculate the sums
            float sum = 0;
            float max = buffer[col];
            for(int i = 0; i < rows; i++) {
                sum += Math.exp(data[col + i * cols] - max);
            }

            // calculate the softmax vectors
            for(int i = 0; i < rows; i++) {
                int index = col + i * cols;
                result.data[index] = (float) (Math.exp(data[index] - max) / sum);
            }
        }

        return result;
    }

//    public Matrix fastBatchSoftmaxDerivative(Matrix output) {
//        Matrix partialDerivatives = new Matrix(cols, cols);
//
//        // for each set of output features
//        for(int outputRow = 0; outputRow < output.rows; outputRow++) {
//            int offset = outputRow * cols;
//            // for each output feature in the set
//            for(int i = 0; i < cols; i++) {
//                float valueI = output.data[offset + i];
//                // for each input feature in the set
//                for(int j = 0; j < cols; j++) {
//                    float valueJ = output.data[offset + j];
//                    if(i == j) {
//                        partialDerivatives.data[i * cols + j] += valueI * (1 - valueI);
//                    }
//                    else {
//                        partialDerivatives.data[i * cols + j] += -valueI * valueJ;
//                    }
//                }
//            }
//        }
//
//        for(int i = 0; i < partialDerivatives.data.length; i++) {
//            partialDerivatives.data[i] /= output.rows;
//        }
//
//        return partialDerivatives;
//    }

    public static boolean isCompatibleWithGPU(GPU gpu) {
        return  gpu.isInitialized() &&
                gpu.getKernel("Matrices::matrixMultiply") != null &&
                gpu.getKernel("Matrices::addRowToRows") != null &&
                gpu.getKernel("Matrices::addColToCols") != null &&
                gpu.getKernel("Matrices::relu") != null &&
                gpu.getKernel("Matrices::horizontalSoftmax") != null &&
                gpu.getKernel("Matrices::verticalSoftmax") != null;
    }

    public Matrix multiply(GPU gpu, Matrix other) {
        if(cols != other.rows) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {other.rows, other.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::matrixMultiply");

        if(kernel == null) {
            throw new NullPointerException("Matrices::matrixMultiply not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, other.cols);

        Pointer pointerA = Pointer.to(data);
        Pointer pointerB = Pointer.to(other.data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryA = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerA, null);
        cl_mem memoryB = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * other.data.length, pointerB, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryA));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryB));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{other.cols}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1, 1};
        long global_work_sizes[] = new long[]{rows, other.cols};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryA);
        clReleaseMemObject(memoryB);
        clReleaseMemObject(memoryOut);

        return result;
    }

    public Matrix addRowToRows(GPU gpu, Matrix row) {
        if(cols != row.cols) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {row.rows, row.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::addRowToRows");

        if(kernel == null) {
            throw new NullPointerException("Matrices::addRowToRows not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, cols);

        Pointer pointerA = Pointer.to(data);
        Pointer pointerB = Pointer.to(row.data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryA = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerA, null);
        cl_mem memoryB = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * row.data.length, pointerB, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryA));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryB));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1};
        long global_work_sizes[] = new long[]{rows};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryA);
        clReleaseMemObject(memoryB);
        clReleaseMemObject(memoryOut);

        return result;
    }

    public Matrix addColToCols(GPU gpu, Matrix col) {
        if(rows != col.rows) {
            final int[] dimensionsA = {rows, cols};
            final int[] dimensionsB = {col.rows, col.cols};
            throw new DimensionsMismatchException(dimensionsA, dimensionsB);
        }

        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::addColToCols");

        if(kernel == null) {
            throw new NullPointerException("Matrices::addColToCols not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, cols);

        Pointer pointerA = Pointer.to(data);
        Pointer pointerB = Pointer.to(col.data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryA = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerA, null);
        cl_mem memoryB = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * col.data.length, pointerB, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryA));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryB));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1};
        long global_work_sizes[] = new long[]{rows};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryA);
        clReleaseMemObject(memoryB);
        clReleaseMemObject(memoryOut);

        return result;
    }

    public Matrix relu(GPU gpu) {
        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::relu");

        if(kernel == null) {
            throw new NullPointerException("Matrices::relu not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, cols);

        Pointer pointerIn = Pointer.to(data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryIn = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerIn, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryIn));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1};
        long global_work_sizes[] = new long[]{rows};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryIn);
        clReleaseMemObject(memoryOut);

        return result;
    }

    public Matrix horizontalSoftmax(GPU gpu) {
        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::horizontalSoftmax");

        if(kernel == null) {
            throw new NullPointerException("Matrices::horizontalSoftmax not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, cols);

        Pointer pointerIn = Pointer.to(data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryIn = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerIn, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryIn));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1};
        long global_work_sizes[] = new long[]{rows};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryIn);
        clReleaseMemObject(memoryOut);

        return result;
    }

    public Matrix verticalSoftmax(GPU gpu) {
        cl_context context = gpu.getContext();
        cl_command_queue commandQueue = gpu.getCommandQueue();
        cl_kernel kernel = gpu.getKernel("Matrices::verticalSoftmax");

        if(kernel == null) {
            throw new NullPointerException("Matrices::verticalSoftmax not found to be loaded in GPU");
        }

        Matrix result = new Matrix(rows, cols);

        Pointer pointerIn = Pointer.to(data);
        Pointer pointerOut = Pointer.to(result.data);

        // Allocate the memory objects for the input- and output data
        cl_mem memoryIn = clCreateBuffer(context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * data.length, pointerIn, null);
        cl_mem memoryOut = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * result.data.length, null, null);

        // Set the arguments for the kernel
        int argNum = 0;
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryOut));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_mem, Pointer.to(memoryIn));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{cols}));
        clSetKernelArg(kernel, argNum++, Sizeof.cl_uint, Pointer.to(new int[]{rows}));

        // Set the work-item dimensions
        long local_work_sizes[] = new long[]{1};
        long global_work_sizes[] = new long[]{cols};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_sizes, local_work_sizes, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, memoryOut, CL_TRUE, 0,
                result.data.length * Sizeof.cl_float, pointerOut, 0, null, null);

        clReleaseMemObject(memoryIn);
        clReleaseMemObject(memoryOut);

        return result;
    }
}
