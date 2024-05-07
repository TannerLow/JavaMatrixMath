package com.github.TannerLow.JavaMatrixMath;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import java.io.Closeable;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateCommandQueueWithProperties;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clGetPlatformInfo;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseProgram;

public class GPU implements Closeable {
    private static final long deviceType = CL_DEVICE_TYPE_GPU;

    private boolean isInitialized = false;
    private cl_platform_id platform;
    private cl_device_id device;
    private cl_context context;
    private cl_command_queue commandQueue;
    private List<cl_program> programs;
    private Map<String, cl_kernel> kernels;

    public GPU() {
        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);
        programs = new ArrayList<>();
        kernels = new HashMap<>();
    }

    public void initialize(boolean automaticSelection) {
        Scanner scanner = null;
        if(!automaticSelection) {
            scanner = new Scanner(System.in);
        }

        // The platform, device type and device number
        // that will be used
        final int defaultPlatformIndex = 0;
        final int defaultDeviceIndex = 0;

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        platform = platforms[defaultPlatformIndex];

        // Prompt for platform selection if automaticSelection is disabled
        if(!automaticSelection) {
            System.out.println("Platforms found:");
            for(int i = 1; i <= numPlatforms; i++) {
                platform = platforms[i-1];
                String platformName = getStringInfo(platform, CL_PLATFORM_NAME);
                System.out.println(i + ".) " + platformName);
            }
            int selection = 0;
            while(selection < 1 || selection > numPlatforms) {
                System.out.print("Select a platform > ");
                selection = scanner.nextInt();
            }
            platform = platforms[selection-1];
        }
        else {
            System.out.println("Automatically selected " + getStringInfo(platform, CL_PLATFORM_NAME));
        }

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        device = devices[defaultDeviceIndex];

        // Prompt for device selection if automaticSelection is disabled
        if(!automaticSelection) {
            System.out.println("Devices found:");
            for(int i = 1; i <= numDevices; i++) {
                device = devices[i-1];
                String deviceName = getStringInfo(device, CL_DEVICE_NAME);
                System.out.println(i + ".) " + deviceName);
            }
            int selection = 0;
            while(selection < 1 || selection > numDevices) {
                System.out.print("Select a device > ");
                selection = scanner.nextInt();
            }
            device = devices[selection-1];
        }
        else {
            System.out.println("Automatically selected " + getStringInfo(device, CL_DEVICE_NAME));
        }

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Create a context for the selected device
        context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
                context, device, properties, null);

        if(!automaticSelection) {
            scanner.close();
        }

        isInitialized = true;
    }

    public int loadProgram(String programCode) throws IllegalStateException {
        if(!isInitialized) {
            throw new IllegalStateException("GPU not yet initialized.");
        }

        // Create the program from the source code
        cl_program program = clCreateProgramWithSource(context,
                1, new String[]{ programCode }, null, null);

        // Build the program
        clBuildProgram(program, 0, null, null, null, null);

        programs.add(program);
        return programs.size()-1;
    }

    public boolean loadKernel(int programId, String scopeName, String kernelName) {
        if(programId >= programs.size() && programId < 0) {
            return false;
        }

        String scopedKernelName = scopeName + "::" + kernelName;

        if(kernels.containsKey(scopedKernelName)) {
            throw new InvalidParameterException("Kernel of that name and scope already exists on GPU.");
        }

        // Create the kernel
        cl_kernel kernel = clCreateKernel(programs.get(programId), kernelName, null);
        kernels.put(scopedKernelName, kernel);

        return true;
    }

    public cl_context getContext() {
        return context;
    }

    public cl_command_queue getCommandQueue() {
        return commandQueue;
    }

    public cl_kernel getKernel(String scopedKernelName) {
        try {
            return kernels.get(scopedKernelName);
        }
        catch(NullPointerException e) {
            return null;
        }
    }

    public boolean isInitialized() {
        return isInitialized;
    }

    @Override
    public void close() {
        isInitialized = false;

        for(Map.Entry<String, cl_kernel> entry : kernels.entrySet()) {
                clReleaseKernel(entry.getValue());
        }

        for(cl_program program : programs) {
            clReleaseProgram(program);
        }

        if(commandQueue != null) {
            clReleaseCommandQueue(commandQueue);
        }

        if(context != null) {
            clReleaseContext(context);
        }
    }

    private static String getStringInfo(cl_platform_id platform, int paramName) {
        long[] size = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1);
    }

    private static String getStringInfo(cl_device_id device, int paramName) {
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);
        byte[] buffer = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length - 1);
    }
}
