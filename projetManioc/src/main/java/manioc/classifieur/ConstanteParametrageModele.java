package manioc.classifieur;

public class ConstanteParametrageModele {
   public String imagesPath ="C:/Users/Joker/Desktop/IF/if5/2021/data science 2021/projet/images/";
   public String trainDataSetPath = imagesPath+ "train_dataset/";
   public String testDataSetPath = imagesPath+ "test_dataset/";
   public String dataSetPath = imagesPath+ "images";
   public String outPut = "C:/Users/Joker/Desktop/IF/if5/2021/data science 2021/projet/label_num_to_disease_map.json";
   public String imagesToPredictClasses = "C:/Users/Joker/Desktop/IF/if5/2021/data science 2021/projet/images_to_predict_classes";
   public String imagesAyantPasClasses = "C:/Users/Joker/Desktop/IF/if5/2021/data science 2021/projet/images_ayant_pas_classes";

    /*
    vitesse d'apprentissage
     */
   public double learningRate = 0.001;
    /*
    Hauteur et largeur des images d'apprentissage
     */
   public long height=60,width=80;
    /*
     depth indique la profondeur de l'image:
     depth = 1 pour les images noir blance
      depth= 3 pour les image en couleur.
     */
   public long depth=3;

}
