package com.github.TannerLow.TestUtils;

public class TestMath {
    public static boolean withinMariginOfError(float a, float b, float marginOfError) {
        float error = Math.abs(a - b);
        return error <= marginOfError;
    }
}
