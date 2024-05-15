package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.TestUtils.TestFailedException;
import com.github.TannerLow.TestUtils.TestMath;

public class CpuTest {
    public static void testAll() {
        testMultiply();
        testAddRowToRows();
        testAddColToCols();
        testRelu();
        testVectorizedReluDerivative();
        testSoftmax();
    }

    private static void testMultiply() {
        float[] aData = {1,2,3,0,1,0};
        float[] bData = {1,0,1,3,2,1};
        float[] expected = {9,9,1,3};

        Matrix a = new Matrix(2,3, aData);
        Matrix b = new Matrix(3,2, bData);

        Matrix result = a.multiply(b);

        if(result.rows != a.rows || result.cols != b.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
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

        Matrix result = a.addRowToRows(b);

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

        Matrix result = a.addColToCols(b);

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

        Matrix result = m.relu();

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }

    private static void testVectorizedReluDerivative() {
        float[] data = {-1,2,-3,0};
        float[] expected = {0,1,0,0};

        Matrix m = new Matrix(2, 2, data);

        Matrix result = m.vectorizedReluDerivative();

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }

    private static void testSoftmax() {
        float[] data = {1.1f,2.2f,0.2f,-1.7f};
        float[] expected = {0.223636f,0.671841f,0.090923f,0.013599f};

        Matrix m = new Matrix(1, 4, data);

        Matrix result = m.softmax();

        if(result.rows != m.rows || result.cols != m.cols) {
            throw new TestFailedException();
        }

        for(int i = 0; i < result.data.length; i++) {
            if(!TestMath.withinMariginOfError(expected[i], result.data[i], 0.0005f)) {
                throw new TestFailedException();
            }
        }
    }
}
