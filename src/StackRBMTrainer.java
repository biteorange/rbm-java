public class StackRBMTrainer {
	RBM layer1; // 784 * 1000
	RBM layer2; // 1000 * 500
	RBM layer3; // 500 * 200
	int maxepoch = 10;
	public StackRBMTrainer(String f1, int visN1, int hidN1,
			String f2, int hidN2, String f3, int hidN3) {
		layer1 = new RBM(f1, visN1, hidN1);
		layer2 = new RBM(f2, hidN1, hidN2);
		layer3 = new RBM(f3, hidN2, hidN3);
	}
	
	public void layerTrain(double[][][] data) {
		int batchSize = data[0].length;
		RBMTrainerBP train1 = new RBMTrainerBP(layer1, maxepoch);
		train1.initialization(layer1.visN, layer1.hidN, batchSize);
	
		// Data data = new Data(600, batchSize, layer1.visN);
		// data.readFromFile("mnist.txt"); 	
	
		train1.stochasticGradientTrain(layer1, data);
		layer1.toFile("rbm1.txt");
		double[][][] data2 = layer1.nextLayerData(data);
		
		// train layer 2
		RBMTrainerBP train2 = new RBMTrainerBP(layer2, maxepoch);
		train2.initialization(layer2.visN, layer2.hidN, batchSize);
		train2.stochasticGradientTrain(layer2, data2);
		layer2.toFile("rbm2.txt");
		
		double[][][] data3 = layer2.nextLayerData(data2);
		
		// train layer 3
		RBMTrainerBP train3 = new RBMTrainerBP(layer3, maxepoch);
		train3.initialization(layer3.visN, layer3.hidN, batchSize);
		train3.stochasticGradientTrain(layer3, data3);
		layer3.toFile("rbm3.txt");
	}
}
