import il2.inf.edgedeletion.*;
import il2.inf.JointEngine;
import il2.inf.PartialDerivativeEngine;
import il2.inf.structure.EliminationOrders;
import il2.model.BayesianNetwork;
import il2.model.Domain;
import il2.model.Table;
import il2.util.IntList;
import il2.util.IntSet;

import il2.inf.Algorithm;
import il2.inf.Algorithm.Setting;
import il2.inf.Algorithm.EliminationOrderHeuristic;
import il2.inf.Algorithm.Order2JoinTree;

import il2.inf.bp.BeliefPropagation;
import il2.inf.bp.MaxProduct;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import edu.ucla.belief.approx.MessagePassingScheduler;


public class RBM {
    // Members
    public BayesianNetwork rbm;
	public Random r; 
	
	// parameters in log spaces
	public double[][] vishidParams;
	public double[] visParams;
	public double[] hidParams;
	public int visN;
	public int hidN;
	
	
	public RBM(String filename, int visN, int hidN) {
		long startTime = System.nanoTime();
		rbm = UaiConverter.uaiToBayesianNetwork(filename);
		System.out.println("finish reading network "+filename+" in "+ (System.nanoTime()-startTime)/1000000000.0);
		this.visN = visN;
		this.hidN = hidN;
		assert(visN+hidN+visN*hidN == rbm.cpts().length);
		
		initializeExpParameters();
	}
	
	public void initializeExpParameters() {
		vishidParams = new double[visN][hidN];
		visParams = new double[visN];
		hidParams = new double[hidN];
		
		Table[] cpts = rbm.cpts();
		for (int i = 0; i < visN; i++) 
			visParams[i] = tabularToExp(cpts[i].values());
		for (int i = 0; i < hidN; i++)
			hidParams[i] = tabularToExp(cpts[visN+i].values());
		System.out.println(visN+" "+hidN);
		for (int i = 0; i < visN; i++) {
			for (int j = 0; j < hidN; j++) {
				vishidParams[i][j] = tabularToExp(cpts[visN+hidN+i*hidN+j].values());
			}
		}
	}
	
	public static void expToTabular(double val, double[] vals) {
		int N = vals.length;
		double exponent = Math.exp(val);
		double norm = N-1 + exponent;
		Arrays.fill(vals, 1.0/norm);
		vals[N-1] = exponent / norm;
	}
	
	public static double tabularToExp(double[] vals) {
		int N = vals.length;
		return Math.log(vals[N-1]/vals[0]);		
	}
	
	public void updateParameters(double[][] vishidinc, double[] visbiasinc, double[] hidbiasinc) {
		Table[] cpts = rbm.cpts();
		
		for (int i = 0; i < visN; i++) {
			for (int j = 0; j < hidN; j++) {
				vishidParams[i][j] += vishidinc[i][j];
				expToTabular(vishidParams[i][j], cpts[visN+hidN+i*hidN+j].values());
			}
		}
		for (int i = 0; i < visN; i++) { 
			visParams[i] += visbiasinc[i];
			expToTabular(visParams[i], cpts[i].values());
		}
		
		for (int j = 0; j < hidN; j++) {
			hidParams[j] += hidbiasinc[j];
			expToTabular(hidParams[j], cpts[j+visN].values());
		}
	}
	
	public void updateBPTables(MixedHuginSSBP ie) {
		Table[] cpts = rbm.cpts();
		for (int var = 0; var < cpts.length; var++) 
			ie.setTable(var, cpts[var]);
	}
	
	/*
	 * RBM layer-to-layer calculations
	 */
	public static double sigmod(double a) {
		return 1.0 / (1.0 + Math.exp(a));
	}
	
	public void visToHid(double[] data, double[] hidVals) {
		assert data.length == visN;
		Arrays.fill(hidVals,  0);
		
		for (int hid = 0; hid < hidN; hid++) {
			for (int vis = 0; vis < visN; vis++) {
				hidVals[hid] -= data[vis]*vishidParams[vis][hid];
			}
			hidVals[hid] -= hidParams[hid];
			hidVals[hid] = sigmod(hidVals[hid]);
		}
	}
	
	public void visToHidBatch(double[][] batch, double[][] hidValsBatch) {
		int N = batch.length;
		assert N == hidValsBatch.length;
		for (int i = 0; i < N; i++) 
			visToHid(batch[i], hidValsBatch[i]);
	}
	
	public void hidToVis(double[] data, double[] visVals) {
		assert data.length == hidN;
		Arrays.fill(visVals,  0);
		
		for (int vis = 0; vis < visN; vis++) {
			for (int hid = 0; hid < hidN; hid++) {
				visVals[vis] -= data[hid]*vishidParams[vis][hid];
			}
			visVals[vis] -= visParams[vis];
			visVals[vis] = sigmod(visVals[vis]);
		}
	}
	
	public void hidToVisBatch(double[][] batch, double[][] visValsBatch) {
		int N = batch.length;
		assert N == visValsBatch.length;
		for (int i = 0; i < N; i++) {
			hidToVis(batch[i], visValsBatch[i]);
		}
	}

	public void binaryStates(double[][] M, double[][] binary) {
		int m = M.length; int n = M[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				binary[i][j] = Math.random() < M[i][j] ? 1 : 0;
			}
		}
	}
	
	public double[][][] nextLayerData(double[][][] batchdata) {
		int nBatches = batchdata.length;
		int batchSize = batchdata[0].length;
		double[][][] nextLayer = new double[nBatches][batchSize][hidN];
		for (int b = 0; b < nBatches; b++)
			visToHidBatch(batchdata[b], nextLayer[b]);
		return nextLayer;
	}
	
	public static void sanityCheck() {
		RBM rbm = new RBM("small.uai", 5, 10);

		RBMTrainerBP rbmTrain = new RBMTrainerBP(rbm, 10);
		// RBMTrainer rbmTrain = new RBMTrainer(10);
		rbmTrain.initialization(5, 10, 3);
		double[][][] batchdata = new double[1][3][5];
		for (int i = 0; i < 3; i++)
			Arrays.fill(batchdata[0][i], 1);
		batchdata[0][0][3] = 0; batchdata[0][0][4] = 0;
		batchdata[0][1][4] = 0;
		rbmTrain.stochasticGradientTrain(rbm, batchdata);
		rbm.toFile("trained.txt");		
		
		StackRBMTrainer stack = new StackRBMTrainer("small1.uai",5,10,"small2.uai",4,"small3.uai",3);
		stack.layerTrain(batchdata);
		
	}
	
	public void toFile(String filename) {
		try {
			PrintStream output = new PrintStream(new File(filename));
			for (int vis = 0; vis < visN; vis++) {
				output.print(visParams[vis]+" ");
			}
			output.println("");
			for (int hid = 0; hid < hidN; hid++) {
				output.print(hidParams[hid]+" ");
			}
			output.println("");
			for (int vis = 0; vis < visN; vis++) {
				for (int hid = 0; hid < hidN; hid++)
					output.print(vishidParams[vis][hid]+" ");
				output.println("");
			}
			output.close();
		}catch (FileNotFoundException e) {
	            e.printStackTrace();
	        }
	}
    
    public static void main(String[] args) {
    	// sanityCheck();
    	
    	
    	int nBatches = 600;
    	int batchSize = 100;
    	Data data = new Data(nBatches, batchSize, 784);
    	data.readFromFile("mnist.txt"); 
    	
    	int layer1 = 1000;
    	int layer2 = 500;
    	int layer3 = 200;
		StackRBMTrainer stack = new StackRBMTrainer("rbm1.uai",784, layer1,"rbm2.uai",layer2,"rbm3.uai",layer3);
		stack.layerTrain(data.getData());
    	
    }
}