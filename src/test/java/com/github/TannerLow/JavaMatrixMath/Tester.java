package com.github.TannerLow.JavaMatrixMath;

import com.github.TannerLow.TestUtils.TestSuiteFailedException;

public class Tester {
    public static void main(String[] args) throws TestSuiteFailedException {
        try {
            CpuTest.testAll();
            GpuTest.testAll();
        }
        catch(Exception e) {
            e.printStackTrace();
            throw new TestSuiteFailedException();
        }

        System.out.println("All tests succeeded");
    }
}
