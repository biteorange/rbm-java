import java.util.Arrays;


public class Algebra {

	// compute M' * V
	public static void transProduct(double[][] M, double[][] V, double[][] res) {
		int m1 = M.length; int n1 = M[0].length;
		int m2 = V.length; int n2 = V[0].length;
		assert m1 == m2;
		assert n1 == res.length;
		assert n2 == res[0].length;
		
		for (int i = 0; i < n1; i++) {
			Arrays.fill(res[i],  0);
		}
		
		for (int i = 0; i < n1; i++) {
			for (int j = 0; j < n2; j++) {
				for (int k = 0; k < m1; k++) {
					res[i][j] += M[k][i]*V[k][j];
				}
			}
		}
	}
	
	// sum rows
	public static void sumRows(double[][] M, double[] rowSum) {
		int m = M.length; int n = M[0].length;
		Arrays.fill(rowSum,  0);
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				rowSum[j] += M[i][j];
	}
}
