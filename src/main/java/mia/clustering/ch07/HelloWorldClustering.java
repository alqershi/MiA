package mia.clustering.ch07;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.kmeans.Cluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class HelloWorldClustering {
    public static final double[][] points = {{1, 1}, {2, 1}, {1, 2},
            {2, 2}, {3, 3}, {8, 8},
            {9, 8}, {8, 9}, {9, 9}};

    public static void writePointsToFile(List<Vector> points,
                                         String fileName,
                                         FileSystem fs,
                                         Configuration conf) throws IOException {
        Path path = new Path(fileName);
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, LongWritable.class, VectorWritable.class);
        long recNum = 0;
        VectorWritable vec = new VectorWritable();
        for (Vector point : points) {
            vec.set(point);
            writer.append(new LongWritable(recNum++), vec);
        }
        writer.close();
    }

    public static List<Vector> getPoints(double[][] raw) {
        List<Vector> points = new ArrayList<Vector>();
        for (double[] fr : raw) {
            Vector vec = new RandomAccessSparseVector(fr.length);
            vec.assign(fr);
            points.add(vec);
        }
        return points;
    }


    public static void main(String args[]) throws Exception {

        int k = 2;

        boolean result = true;

        List<Vector> vectors = getPoints(points);

        File testData = new File("/tmp/testdata");

        if (!testData.exists()) {
            result = testData.mkdir();
        }

        testData = new File("/tmp/testdata/points");

        if (!testData.exists()) {
            result = testData.mkdir();
        }

        if(!result)
        {
            System.out.println("Kindly check that you have set up the environment variables in your project or IDE");
        }

        for (Vector vector : vectors) {
            System.out.println("vector point: " + vector);
        }

        Configuration conf = new Configuration();

        FileSystem fs = FileSystem.get(conf);

        writePointsToFile(vectors, "/tmp/testdata/points/file1", fs, conf);

        Path path = new Path("/tmp/testdata/clusters/part-00000");
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
                path, Text.class, Cluster.class);

        for (int i = 0; i < k; i++) {
            Vector vec = vectors.get(i);
            Cluster cluster = new Cluster(vec, i, new EuclideanDistanceMeasure());
            System.out.println(new Text(cluster.getIdentifier()) + "," + cluster);
            writer.append(new Text(cluster.getIdentifier()), cluster);
        }
        writer.close();

    }
}
