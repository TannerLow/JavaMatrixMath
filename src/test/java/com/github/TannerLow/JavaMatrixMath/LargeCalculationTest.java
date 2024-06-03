package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.TestUtils.TestFailedException;
import com.github.TannerLow.TestUtils.TestMath;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class LargeCalculationTest {
    private static final int SIZE = 1500;
    private static final GPU gpu = new GPU();

    public static void main(String[] args) throws IOException {
        try(gpu) {
            setup();

            long startTime = System.currentTimeMillis();
            testSmallMultiplyGPU();
            long duration = System.currentTimeMillis() - startTime;
            System.out.println("Time to run small multiplication on GPU: " + duration + "ms");

            startTime = System.currentTimeMillis();
            testSmallMultiplyCPU();
            duration = System.currentTimeMillis() - startTime;
            System.out.println("Time to run small multiplication on CPU: " + duration + "ms");

            startTime = System.currentTimeMillis();
            testLargeMultiplyGPU();
            duration = System.currentTimeMillis() - startTime;
            System.out.println("Time to run large multiplication on GPU: " + duration + "ms");

            startTime = System.currentTimeMillis();
            testLargeMultiplyCPU();
            duration = System.currentTimeMillis() - startTime;
            System.out.println("Time to run large multiplication on CPU: " + duration + "ms");
        }
    }

    private static void setup() throws IOException {
        // Load GPU program code into memory
        String matricesKernelFilePath = "kernels/Matrices.cl";
        String matricesKernelCode = readFromInternalFile(matricesKernelFilePath);
        if(matricesKernelCode == null) {
            throw new IOException("Failed to read file: " + matricesKernelFilePath);
        }

        gpu.initialize(true);
        int programId = gpu.loadProgram(matricesKernelCode);
        gpu.loadKernel(programId, "Matrices", "matrixMultiply");
        gpu.loadKernel(programId, "Matrices", "addRowToRows");
        gpu.loadKernel(programId, "Matrices", "addColToCols");
        gpu.loadKernel(programId, "Matrices", "relu");
        gpu.loadKernel(programId, "Matrices", "horizontalSoftmax");
        gpu.loadKernel(programId, "Matrices", "verticalSoftmax");

        if(!gpu.isInitialized() || !Matrix.isCompatibleWithGPU(gpu)) {
            throw new IllegalStateException("GPU in unexpected state.");
        }
    }

    private static void testSmallMultiplyGPU() {
        float[] aData = {1,2,3,0,1,0};
        float[] bData = {1,0,1,3,2,1};

        Matrix a = new Matrix(2,3, aData);
        Matrix b = new Matrix(3,2, bData);

        Matrix result = a.multiply(gpu, b);
    }

    private static void testSmallMultiplyCPU() {
        float[] aData = {1,2,3,0,1,0};
        float[] bData = {1,0,1,3,2,1};

        Matrix a = new Matrix(2,3, aData);
        Matrix b = new Matrix(3,2, bData);

        Matrix result = a.multiply(b);
    }

    private static void testLargeMultiplyGPU() {
        float[] aData = new float[SIZE*SIZE];
        float[] bData = new float[SIZE*SIZE];

        Matrix a = new Matrix(SIZE, SIZE, aData);
        Matrix b = new Matrix(SIZE, SIZE, bData);

        Matrix result = a.multiply(gpu, b);
    }

    private static void testLargeMultiplyCPU() {
        float[] aData = new float[SIZE*SIZE];
        float[] bData = new float[SIZE*SIZE];

        Matrix a = new Matrix(SIZE, SIZE, aData);
        Matrix b = new Matrix(SIZE, SIZE, bData);

        Matrix result = a.multiply(b);
    }

    // Helper
    private static String readFromInternalFile(String filepath) {
        try(InputStream fileInputStream = InternalFile.getInstance().getFileInputStream(filepath)) {
            byte[] bytes = fileInputStream.readAllBytes();
            String fileContent = new String(bytes, StandardCharsets.UTF_8);
            return fileContent;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }
}
