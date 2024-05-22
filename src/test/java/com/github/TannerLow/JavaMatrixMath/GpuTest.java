package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.TestUtils.TestFailedException;
import com.github.TannerLow.TestUtils.TestMath;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class GpuTest {

    private static final GPU gpu = new GPU();

    public static void testAll() throws IOException {
        try(gpu) {
            setup();

            testMultiply();
            testAddRowToRows();
            testAddColToCols();
            testRelu();
            testHorizontalSoftmax();
            testVerticalSoftmax();
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

    private static void testMultiply() {
        float[] aData = {1,2,3,0,1,0};
        float[] bData = {1,0,1,3,2,1};
        float[] expected = {9,9,1,3};

        Matrix a = new Matrix(2,3, aData);
        Matrix b = new Matrix(3,2, bData);

        Matrix result = a.multiply(gpu, b);

        if(result.rows != a.rows || result.cols != b.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                System.out.println(expected[i] + " vs " + result.data[i]);
                throw new TestFailedException();
            }
        }
    }

    private static void testAddRowToRows() {
        float[] aData = {1,2,3,0,0,0};
        float[] bData = {3,2,1};
        float[] expected = {4,4,4,3,2,1};

        Matrix a = new Matrix(2,3, aData);
        Matrix b = new Matrix(1,3, bData);

        Matrix result = a.addRowToRows(gpu, b);

        if(result.rows != a.rows || result.cols != a.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }

    private static void testAddColToCols() {
        float[] aData = {1,0,2,0,3,0};
        float[] bData = {3,2,1};
        float[] expected = {4,3,4,2,4,1};

        Matrix a = new Matrix(3,2, aData);
        Matrix b = new Matrix(3,1, bData);

        Matrix result = a.addColToCols(gpu, b);

        if(result.rows != a.rows || result.cols != a.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }

    private static void testRelu() {
        float[] data = {-1,2,-3,0};
        float[] expected = {0,2,0,0};

        Matrix m = new Matrix(2, 2, data);

        Matrix result = m.relu(gpu);

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }

    private static void testHorizontalSoftmax() {
        float[] data = {1.1f,2.2f,0.2f,-1.7f};
        float[] expected = {0.223636f,0.671841f,0.090923f,0.013599f};

        Matrix m = new Matrix(1, 4, data);

        Matrix result = m.horizontalSoftmax(gpu);

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                System.out.println(expected[i] + " vs. " + result.data[i]);
                throw new TestFailedException();
            }
        }
    }

    private static void testVerticalSoftmax() {
        float[] data = {1.1f,2.2f,0.2f,-1.7f};
        float[] expected = {0.223636f,0.671841f,0.090923f,0.013599f};

        Matrix m = new Matrix(4, 1, data);

        Matrix result = m.verticalSoftmax(gpu);

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                System.out.println(expected[i] + " vs. " + result.data[i]);
                throw new TestFailedException();
            }
        }
    }

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
