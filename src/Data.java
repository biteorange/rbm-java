import java.io.File;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;

// read mnist data from matlab
public class Data {
	public double[][][] batchdata;
	public int batches;
	public int cases;
	public int dim;
	
	public Data(int batches, int cases, int dim) {
		batchdata = new double[batches][cases][dim];
		this.batches = batches;
		this.cases = cases;
		this.dim = dim;
	}
	
	public Data(double[][][] batchdata) {
		this.batchdata = batchdata;
		this.batches = batchdata.length;
		this.cases = batchdata[0].length;
		this.dim = batchdata[0][0].length;
	}
	
	public double[][][] getData() {
		return batchdata;
	}
	public double[][] getBatch(int b) {
		return batchdata[b];
	}
	
	public void readFromFile(String filename) {
		Scanner sc;
		try {
			sc = new Scanner(new File(filename));
		} catch (FileNotFoundException e) {
			System.out.println(filename+" not found");
			return;
		}

		for (int b = 0; b < batches; b++) {
			for (int n = 0; n < cases; n++) {
				for (int i = 0; i < dim; i++) 
					batchdata[b][n][i] = sc.nextDouble();
			}
			System.out.println("finish batch "+b);
		}
		sc.close();
	}
}
