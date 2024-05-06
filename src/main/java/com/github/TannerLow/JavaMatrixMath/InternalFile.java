package com.github.TannerLow.JavaMatrixMath;

import java.io.InputStream;


public class InternalFile {

    private static final InternalFile INSTANCE = new InternalFile();

    private InternalFile() {}

    public static InternalFile getInstance() {
        return INSTANCE;
    }

    public InputStream getFileInputStream(String filePath) throws NullPointerException{
        return getClass().getClassLoader().getResourceAsStream(filePath);
    }
}
