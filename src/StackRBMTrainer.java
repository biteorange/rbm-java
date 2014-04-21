public class StackRBMTrainer {
	RBM layer1; // 784 * 1000
	RBM layer2; // 1000 * 500
	RBM layer3; // 500 * 200
	int maxepoch = 10;
	int batchSize = 100;
	public StackRBMTrainer(String f1, String f2, String f3) {
		layer1 = new RBM(f1, 784, 1000);
		layer2 = new RBM(f2, 1000, 500);
		layer3 = new RBM(f3, 500, 200);
	}
	
	public void layerTrain() {
		RBMTrainerBP train1 = new RBMTrainerBP(layer1, maxepoch);
		train1.initialization(layer1.visN, layer1.hidN, batchSize);
	
		Data data = new Data(600, batchSize, layer1.visN);
		data.readFromFile("mnist.txt"); 	
	
		train1.stochasticGradientTrain(layer1, data.getData());
		double[][][] data2 = layer1.nextLayerData(data.getData());
		
		// train layer 2
		RBMTrainerBP train2 = new RBMTrainerBP(layer2, maxepoch);
		train2.initialization(layer2.visN, layer2.hidN, batchSize);
		train2.stochasticGradientTrain(layer2, data2);
		
		double[][][] data3 = layer2.nextLayerData(data2);
		
		// train layer 3
		RBMTrainerBP train3 = new RBMTrainerBP(layer3, maxepoch);
		train3.initialization(layer3.visN, layer3.hidN, batchSize);
		train3.stochasticGradientTrain(layer3, data3);
	}
}
