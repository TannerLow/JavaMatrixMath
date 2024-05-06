package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.TestUtils.TestFailedException;
import com.github.TannerLow.TestUtils.TestMath;

public class CpuTest {
    public static void testAll() {
        testMultiply();
        testAddRowToRows();
        testRelu();
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
}
