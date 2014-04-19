import il2.model.Table;


public class RBMTrainerBP extends RBMTrainer{
    // ED-BP (RCR) settings:
	public int    maxIterations = 1;
	public long   timeoutMillis = 0;    //no time-out
	public double convThreshold = 1e-4;
	
	MixedHuginSSBP ie;
	
	public RBMTrainerBP(RBM rbm, int maxepoch) {
		super(maxepoch);
		
		double startTime = System.nanoTime();
		ie = startBPInferenceEngine(rbm.rbm.cpts());
        System.out.println("starting engine cost "+ (System.nanoTime()-startTime)/1000000000.0);
	}
	
	public MixedHuginSSBP startBPInferenceEngine(Table[] cpts) {
		return new MixedHuginSSBP(cpts, this.maxIterations, this.timeoutMillis, this.convThreshold);
	}
	
	@Override
	public void computeGradientNeg(RBM rbm, double[][] data) {
		int N = data.length;
		int visN = rbm.visN; int hidN = rbm.hidN;
		for (int i = 0; i < visN; i++) {
			negvisact[i] = N * last(ie.tableConditional(i).values());
		}
		for (int j = 0; j < hidN; j++) {
			neghidact[j] = N * last(ie.tableConditional(visN+j).values());
		}
		for (int i = 0; i < visN; i++) {
			for (int j = 0; j < hidN; j++) {
				negprobs[i][j] = N * last(ie.tableConditional(visN+hidN+i*visN+j).values());
			}
		}
	}
	
	public double last(double[] vals) {
		return vals[vals.length-1];
	}
	
	@Override
	public void trainBatch(RBM rbm, double[][] batch) {
		super.trainBatch(rbm, batch);
		rbm.updateBPTables(ie);
	}
}
