/*
 * Copyright 2016 Mario Contreras <marioc@nazul.net>.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 *
 * @author Mario Contreras <marioc@nazul.net>
 */
package mx.iteso.msc.examples;

import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.imagebundle.mapreduce.HibInputFormat;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;

import java.io.IOException;
import java.net.URI;

public class FaceCount extends Configured implements Tool {

    public static class FaceCountMapper extends Mapper<HipiImageHeader, FloatImage, IntWritable, IntWritable> {

        // Create a face detector from the cascade file in the resources
        // directory.
        private CascadeClassifier faceDetector;

        // Convert HIPI FloatImage to OpenCV Mat
        public Mat convertFloatImageToOpenCVMat(FloatImage floatImage) {

            // Get dimensions of image
            int w = floatImage.getWidth();
            int h = floatImage.getHeight();

            // Get pointer to image data
            float[] valData = floatImage.getData();

            // Initialize 3 element array to hold RGB pixel average
            double[] rgb = {0.0, 0.0, 0.0};

            Mat mat = new Mat(h, w, CvType.CV_8UC3);

            // Traverse image pixel data in raster-scan order and update running average
            for (int j = 0; j < h; j++) {
                for (int i = 0; i < w; i++) {
                    rgb[0] = (double) valData[(j * w + i) * 3 + 0] * 255.0; // R
                    rgb[1] = (double) valData[(j * w + i) * 3 + 1] * 255.0; // G
                    rgb[2] = (double) valData[(j * w + i) * 3 + 2] * 255.0; // B
                    mat.put(j, i, rgb);
                }
            }

            return mat;
        }

        // Count faces in image
        public int countFaces(Mat image) {

            // Detect faces in the image.
            // MatOfRect is a special container class for Rect.
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(image, faceDetections);

            return faceDetections.toArray().length;
        }

        public void setup(Context context)
                throws IOException, InterruptedException {

            // Load OpenCV native library
            try {
                System.load("/usr/local/share/OpenCV/java/libopencv_java2413.so");
                //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                System.out.println("Native library loaded successfully");
            } catch (UnsatisfiedLinkError e) {
                System.err.println("Native code library failed to load ---- .\n" + e + Core.NATIVE_LIBRARY_NAME);
                System.exit(1);
            }

            // Load cached cascade file for front face detection and create CascadeClassifier
            if (context.getCacheFiles() != null && context.getCacheFiles().length > 0) {
                URI mappingFileUri = context.getCacheFiles()[0];

                if (mappingFileUri != null) {
                    faceDetector = new CascadeClassifier("lbpcascade_frontalface.xml");

                } else {
                    System.err.println(">>>>>> NO MAPPING FILE");
                }
            } else {
                System.err.println(">>>>>> NO CACHE FILES AT ALL");
            }

            super.setup(context);
        } // setup()

        public void map(HipiImageHeader key, FloatImage value, Context context)
                throws IOException, InterruptedException {

            // Verify that image was properly decoded, is of sufficient size, and has three color channels (RGB)
            //if (value != null && value.getWidth() > 1 && value.getHeight() > 1 && value.getBands() == 3) {
            if (value != null && value.getWidth() > 1 && value.getHeight() > 1) {

                Mat cvImage = this.convertFloatImageToOpenCVMat(value);

                int faces = this.countFaces(cvImage);

                System.out.println(">>>>>> Detected Faces: " + Integer.toString(faces));

                // Emit record to reducer
                context.write(new IntWritable(1), new IntWritable(faces));

            } // If (value != null...

        } // map()
    }

    public static class FaceCountReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            // Initialize a counter and iterate over IntWritable/FloatImage records from mapper
            int total = 0;
            int images = 0;
            for (IntWritable val : values) {
                total += val.get();
                images++;
            }

            String result = String.format("Total face detected: %d", total);
            // Emit output of job which will be written to HDFS
            context.write(new IntWritable(images), new Text(result));
        } // reduce()
    }

    public int run(String[] args) throws Exception {
        // Check input arguments
        if (args.length != 2) {
            System.out.println("Usage: FaceCount <input HIB> <output directory>");
            System.exit(0);
        }

        // Initialize and configure MapReduce job
        Job job = Job.getInstance();
        // Set input format class which parses the input HIB and spawns map tasks
        job.setInputFormatClass(HibInputFormat.class);
        // Set the driver, mapper, and reducer classes which express the computation
        job.setJarByClass(FaceCount.class);
        job.setMapperClass(FaceCountMapper.class);
        job.setReducerClass(FaceCountReducer.class);
        // Set the types for the key/value pairs passed to/from map and reduce layers
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        // Set the input and output paths on the HDFS
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // add cascade file
        job.addCacheFile(new URI("/hipi/OpenCV/lbpcascade_frontalface.xml#lbpcascade_frontalface.xml"));

        // Execute the MapReduce job and block until it complets
        boolean success = job.waitForCompletion(true);

        // Return success or failure
        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new FaceCount(), args);
        System.exit(0);
    }
}

//EOF
