import java.util.Arrays;


public class RBMTrainer {

	// sgd parameters
	protected double epsilonw = 0.1;
	protected double epsilonvb = 0.1;
	protected double epsilonhb = 0.1;
	protected double weightcost = 0.0002;
	protected double initialmomentum = 0.5;
	protected double finalmomentum = 0.9;
	protected double momentum = initialmomentum;
	protected int maxepoch = 0;
	
	// gradient increments
	protected double[][] vishidinc;
	protected double[] visbiasinc;
	protected double[] hidbiasinc;
	
	// training utilities
	double[][] poshidprobs;
	double[][] neghidprobs;
	double[][] posprods;
	double[][] negprobs;
	double[] poshidact;
	double[] posvisact;
	double[] neghidact;
	double[] negvisact;
	
	// CD statistics
	double[][] poshidstates;
	double[][] negdata;
	
	
	public RBMTrainer(int maxepoch) {
		this.maxepoch = maxepoch;
	}
	
	public double[][] zeroMatrix(int m, int n) {
		double[][] matrix = new double[m][n];
		return matrix;
	}
	
	public double[] zeroVector(int m) {
		double[] vector = new double[m];
		return vector;
	}
	
	public void initialization(int visN, int hidN, int N) {
		vishidinc = zeroMatrix(visN, hidN);
		visbiasinc = zeroVector(visN);
		hidbiasinc = zeroVector(hidN);
		
		poshidprobs = zeroMatrix(N, hidN);
		neghidprobs = zeroMatrix(N, hidN);
		posprods = zeroMatrix(visN, hidN);
		negprobs = zeroMatrix(visN, hidN);
		poshidact = zeroVector(hidN);
		posvisact = zeroVector(visN);
		neghidact = zeroVector(hidN);
		negvisact = zeroVector(visN);
		
		poshidstates = zeroMatrix(N, hidN);
		negdata = zeroMatrix(N, visN);
	}
	
	public void computeGradientPos(RBM rbm, double[][] batch) {
		rbm.visToHidBatch(batch, poshidprobs);
		Algebra.transProduct(batch, poshidprobs, posprods);
		Algebra.sumRows(poshidprobs, poshidact);
		Algebra.sumRows(batch, posvisact);
	}
	
	public void computeGradientNeg(RBM rbm, double[][] batch) {
		rbm.binaryStates(poshidprobs, poshidstates);
		rbm.hidToVisBatch(poshidstates, negdata);
		rbm.visToHidBatch(negdata, neghidprobs);
		Algebra.transProduct(negdata, neghidprobs, negprobs);
		Algebra.sumRows(neghidprobs, neghidact);
		Algebra.sumRows(negdata, negvisact);
	}
	
	
	public void trainBatch(RBM rbm, double[][] batch) {
		int N = batch.length;
		computeGradientPos(rbm, batch);
		computeGradientNeg(rbm, batch);

		for (int i = 0; i < rbm.visN; i++) {
			for (int j = 0; j < rbm.hidN; j++) {
				vishidinc[i][j] = momentum * vishidinc[i][j] + 
						epsilonw*((posprods[i][j]-negprobs[i][j])/N - weightcost*rbm.vishidParams[i][j]);
			}
		}
		
		for (int i = 0; i < rbm.visN; i++)
			visbiasinc[i] = momentum*visbiasinc[i] + (epsilonvb/N)*(posvisact[i]-negvisact[i]);
		
		for (int j = 0; j < rbm.hidN; j++)
			hidbiasinc[j] = momentum*hidbiasinc[j] + (epsilonhb/N)*(poshidact[j]-neghidact[j]);
		
		rbm.updateParameters(vishidinc, visbiasinc, hidbiasinc);
	}
	
	public void stochasticGradientTrain(RBM rbm, double[][][] data) {
		int nBatches = data.length;
		for (int epoch = 0; epoch < maxepoch; epoch++) {
			if (epoch > 5)
				momentum = finalmomentum;
			else
				momentum = initialmomentum;
			for (int b = 0; b < nBatches; b++) {
				double start = System.nanoTime();
				trainBatch(rbm, data[b]);	
				double time = (System.nanoTime()-start)/1000000000.0;
				System.out.println("epoch "+epoch+" batch "+b+" "+time+" secs");
			}
		}
	}
}
