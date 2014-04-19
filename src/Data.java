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
	
	public double[][] getBatch(int b) {
		return batchdata[b];
	}
	
	public void readFromFile(String filename) {
		Scanner sc = new Scanner(filename);
		for (int b = 0; b < batches; b++) {
			for (int n = 0; n < cases; n++) {
				for (int i = 0; i < dim; i++) 
					batchdata[b][n][i] = sc.nextDouble();
			}
		}
		sc.close();
	}
}
