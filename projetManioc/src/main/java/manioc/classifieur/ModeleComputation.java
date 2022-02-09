package manioc.classifieur;

import java.io.File;
import java.io.IOException;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModeleComputation {
	
	public MultiLayerConfiguration configurationModele() {
		
		ConstanteParametrageModele constanteParam = new ConstanteParametrageModele();
		
        //configuration du modèle
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(1234) // pour génerer le même nombre aléatoire Ã  chaque exécution du modèle
                .updater(new Sgd(constanteParam.learningRate)) //algorithme de propagation du gradiant
                .list()
                .setInputType(InputType.convolutionalFlat(constanteParam.height,constanteParam.width,constanteParam.depth))// indique que le model apprendre sur les imamges
                .layer(0,new ConvolutionLayer.Builder()//couche de convolution
                        .nIn(constanteParam.depth)
                        .nOut(20)// le nombre d'image à  la sotie après le filtrage
                        .activation(Activation.RELU) //fonction d'activation = 0 si pixel <0 et 1 si non
                        .kernelSize(5,5) //filtre: une matrice de 5*5
                        .stride(1,1) // fenÃ¨tre de glissement du filtre dans l'image original: 1,1 => 1 pixel vertical et 1 pixel horizontal
                        .build())
                .layer(1,new SubsamplingLayer.Builder() //couche de max pulling
                        .kernelSize(2,2)//chaque 2*2 pixel, garder que le maximum de pixel
                        .stride(2,2) //fenÃ¨tre de glissement
                        .poolingType(SubsamplingLayer.PoolingType.MAX)// type de pulling: maximum de pixel, on peut utiliser d'autre fonction comme moyenne, somme,...
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)
                        .stride(1,1)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())                
                
                .layer(4,new DenseLayer.Builder() //couche fully connected
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(5,new OutputLayer.Builder()
                        .nOut(5)
                        .activation(Activation.SOFTMAX)//on utilise softmax pour la somme des proba donne 1 (proba pour que la sortie du modelsoit 0,1,2,3,4)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)// minimiser les erreures
                        .build()
                )
                .build();
        
        //System.out.println(configuration.toJson());
        return configuration;
	}

	public MultiLayerNetwork entrainementModele() throws IOException, InterruptedException {
		ConstanteParametrageModele constanteParam = new ConstanteParametrageModele();

        /*
        	instancier le modèle
         */
       MultiLayerNetwork modele = new MultiLayerNetwork(configurationModele());
        modele.init();

        File fileTrain = new File(constanteParam.trainDataSetPath);
        FileSplit fileSplit = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS); //lecture du contenu du filTrain (i.e les images
        // lire les images à  partir de fileSplit
        RecordReader recordReaderTrain = new ImageRecordReader(constanteParam.height,constanteParam.width,constanteParam.depth,new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplit);
        //iterer sur les images avec un batch de 400: nombre d'image lue en groupe et presenter au modele
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReaderTrain,400,1,5);
        DataNormalization dataNormalization = new ImagePreProcessingScaler(0,1);
        dataSetIterator.setPreProcessor(dataNormalization);
        /*
        loguer l'apprentissage
         */
         UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        modele.setListeners(new StatsListener(statsStorage));

        int numEpoch =100;// est un exemple de nombre d'iteration permettant au modele de bien apprendre
        for(int i =0;i<numEpoch;i++)
            modele.fit(dataSetIterator);//c'est la fonction d'apprendtissage.
                
        return modele;
	}

	public void testerModele() throws IOException, InterruptedException {
		
		ConstanteParametrageModele constanteParam = new ConstanteParametrageModele();

		MultiLayerNetwork modele = entrainementModele();
        /*
         * lecture des images de test
         */
        File fileTest = new File(constanteParam.testDataSetPath);
        FileSplit testFileSplit = new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS); //lecture du contenu du fileTest (i.e les images
        
        // lire les images à  partir de testFileSplit
        RecordReader recordReaderTest = new ImageRecordReader(constanteParam.height,constanteParam.width,constanteParam.depth,new ParentPathLabelGenerator());
        recordReaderTest.initialize(testFileSplit);
        //iterer sur les images avec un batch de 400: nombre d'image lue en groupe et presenter au modele
        DataSetIterator testDataSetIterator = new RecordReaderDataSetIterator(recordReaderTest,400,1,5);
        DataNormalization testDataNormalization = new ImagePreProcessingScaler(0,1);
        testDataSetIterator.setPreProcessor(testDataNormalization);
        Evaluation evaluation = new Evaluation();

        /*
         * Evaluation du modèle
         */
        while(testDataSetIterator.hasNext())
        {
            DataSet dataSet = testDataSetIterator.next();
            INDArray features= dataSet.getFeatures();
            INDArray targetLabels = dataSet.getLabels();
            INDArray precditLabels = modele.output(features);
            evaluation.eval(precditLabels,targetLabels);

        }
        /*
         * logger les metrics comme matrice confusion, accuracy, precision,...
         */
        System.out.println(evaluation.stats());

        //enregistrer le modele en format .zip
        ModelSerializer.writeModel(modele,"maniocModele.zip",true);
   }
	
	public INDArray prediction() throws IOException, InterruptedException {
		ConstanteParametrageModele constanteParam = new ConstanteParametrageModele();
		//restorer le modèle
        MultiLayerNetwork modele = ModelSerializer.restoreMultiLayerNetwork(new File("maniocModele.zip"));
		
		/*
         * lecture des images à predire leur classe
         */
        File filePredict = new File(constanteParam.imagesToPredictClasses);
        FileSplit predictFileSplit = new FileSplit(filePredict, NativeImageLoader.ALLOWED_FORMATS); //lecture du contenu du fileTest (i.e les images
         // lire les images à  partir de testFileSplit
        RecordReader recordReaderPredict = new ImageRecordReader(constanteParam.height,constanteParam.width,constanteParam.depth,new ParentPathLabelGenerator());
        recordReaderPredict.initialize(predictFileSplit);
        //iterer sur les images avec un batch de 400: nombre d'image lue en groupe et presenter au modele
        DataSetIterator predictDataSetIterator = new RecordReaderDataSetIterator(recordReaderPredict,400,1,5);
        DataNormalization predictDataNormalization = new ImagePreProcessingScaler(0,1);
        predictDataSetIterator.setPreProcessor(predictDataNormalization);
        Evaluation evaluation = new Evaluation();
        DataSet dataSet = predictDataSetIterator.next();
        INDArray precditLabels = modele.output(dataSet.getFeatures());	
        //modele.predict(dataSet)
		return precditLabels;
	}

}
