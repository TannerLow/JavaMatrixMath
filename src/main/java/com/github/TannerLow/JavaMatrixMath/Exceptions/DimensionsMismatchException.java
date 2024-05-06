package com.github.TannerLow.JavaMatrixMath.Exceptions;

public class DimensionsMismatchException extends IllegalArgumentException {

    private int[] dimensionsA;
    private int[] dimensionsB;

    public DimensionsMismatchException(int[] dimensionsA, int[] dimensionsB) {
        super();
        this.dimensionsA = dimensionsA;
        this.dimensionsB = dimensionsB;
    }

    @Override
    public String getMessage() {
        String dimensionsAString = dimensionsToString(this.dimensionsA);
        String dimensionsBString = dimensionsToString(this.dimensionsB);
        return "Dimensions mismatch: " + dimensionsAString + " " + dimensionsBString;
    }

    private static String dimensionsToString(final int[] dimensions) {
        if(dimensions == null || dimensions.length == 0) {
            return "(0)";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("(");
        sb.append(dimensions[0]);
        for(int i = 1; i < dimensions.length; i++) {
            sb.append("x");
            sb.append(dimensions[i]);
        }
        sb.append(")");

        return sb.toString();
    }
}
