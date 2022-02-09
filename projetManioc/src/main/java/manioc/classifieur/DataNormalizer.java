package manioc.classifieur;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.opencsv.CSVReader;
import lombok.NoArgsConstructor;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

@NoArgsConstructor
public class DataNormalizer {
	ConstanteParametrageModele constanteParametrageModele = new ConstanteParametrageModele();
    private Map<String, String> trainLabels = new HashMap<>(); //ensembles des couples nom image et la sortie de l'image
    private Set<String> labels = new HashSet<>();// nom des sortie
    private Map<String, String> output = new HashMap<>(); //ensemble des couple sortie et nom maladie ou decision

    public void preprocessingData() throws IOException {
        prepareTrainLabels();
        createDirectories(constanteParametrageModele.imagesPath+"train_dataset/");
        createDirectories(constanteParametrageModele.imagesPath+"test_dataset/");
        dataDispatcher();
    }

    /*
    lire le fichier csv train.csv contenant le nom de chaque image et sa classe.
     */
    private void prepareTrainLabels(){
        CSVReader reader = null;
        try {
            reader = new CSVReader(
                    new FileReader(
                            "C:/Users/Joker/Desktop/IF/if5/2021/data science 2021/projet/image_to_label_num_map.csv"));
            String[] line;
            reader.readNext();//supprimer l'entÃªte
            while ((line= reader.readNext())!= null){
                trainLabels.put(line[0],line[1]);
                labels.add(line[1]);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    /*
    crÃ©er les repertoire de chaque classe
     */
    private void createDirectories(String directoryPath){
        for( String label: labels){
            File f = new File(directoryPath+label);
            f.mkdir();
        }
    }
    private void dataDispatcher() throws IOException {
        File file;
        file = new File(constanteParametrageModele.dataSetPath);
        String[] imageNames = file.list();
        for (int i = 0; i < imageNames.length; i++) {
        	if(trainLabels.get(imageNames[i])!= null) {
                if(Files.notExists(Paths.get(constanteParametrageModele.trainDataSetPath + trainLabels.get(imageNames[i])+"/"+imageNames[i]))){
                    BufferedImage image = ImageIO.read(new File(constanteParametrageModele.dataSetPath+"/" + imageNames[i]));
                    Image image1 = image.getScaledInstance(80, 60, Image.SCALE_SMOOTH);
                    BufferedImage buffered = new BufferedImage(80, 60, BufferedImage.TYPE_3BYTE_BGR);
                    buffered.getGraphics().drawImage(image1, 0, 0 , null);
                    ImageIO.write( buffered, "jpg",
                            new File(constanteParametrageModele.trainDataSetPath + trainLabels.get(imageNames[i])+"/"+imageNames[i]));
                    Files.deleteIfExists(Paths.get(constanteParametrageModele.dataSetPath  + "/" + imageNames[i]));
                }        		
        	}else {
        		if(Files.notExists(Paths.get(new ConstanteParametrageModele().imagesAyantPasClasses+"/"+imageNames[i]))){
                    ImageIO.write( ImageIO.read(new File(constanteParametrageModele.dataSetPath+"/" + imageNames[i])), "jpg",
                            new File(new ConstanteParametrageModele().imagesAyantPasClasses+"/"+imageNames[i]));        			
        		}
                Files.deleteIfExists(Paths.get(constanteParametrageModele.dataSetPath  + "/" + imageNames[i]));
        	}
        }

       preprareTestData();
    }

    /*
    Preparer les données de test
     */
    private void preprareTestData() throws IOException {
        File file;
        List<String> list;
        for (String label: labels) {
            file = new File(constanteParametrageModele.trainDataSetPath+label+"/");
            String[] imageNames = file.list();
            list = new ArrayList<>();
            for(String imageName: imageNames){
                list.add(imageName);
            }
            int testDataSize = (int) (0.2* list.size());            
            Random ran = new Random();
            for (int i = 0; i < testDataSize ; i++) {
               int indexImage = ran.nextInt(list.size());                
               if(testDataSize > new File(constanteParametrageModele.testDataSetPath+label+"/").list().length){
                   ImageIO.write( ImageIO.read(new File(constanteParametrageModele.trainDataSetPath + label + "/" + list.get(indexImage))),
                      "jpg", new File(constanteParametrageModele.testDataSetPath+label+"/" + list.get(indexImage)));
                      Files.deleteIfExists(Paths.get(constanteParametrageModele.trainDataSetPath + label + "/" + list.get(indexImage)));
                    list.remove(indexImage);
                }else {
                	break;
                }
            }
          }
    }
    
    /*
     * Redimensionner l'image
     */
    public void redimensionneImage(String imagePath) throws IOException {
    	BufferedImage image = ImageIO.read(new File(imagePath));
        Image image1 = image.getScaledInstance(80, 60, Image.SCALE_SMOOTH);
        BufferedImage buffered = new BufferedImage(80, 60, BufferedImage.TYPE_3BYTE_BGR);
        buffered.getGraphics().drawImage(image1, 0, 0 , null);
        ImageIO.write( buffered, "jpg",new File(imagePath+"/"+ new File(imagePath).list()[0]));                   
    }

    public Map<String, String> readOutputFromJSONFile(String outPutFilePath) throws JsonProcessingException {
        //JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
        
        try (FileReader reader = new FileReader(outPutFilePath))
        {
            //Read JSON file
            Object obj = jsonParser.parse(reader);
            JSONObject outputContent = (JSONObject) obj;
            Map<String, String> output = new HashMap<>();
            for(String label: labels){
                output.put(label,(String) outputContent.get(label));
            }
            return output;
        } catch (Exception e) {
        	return null;
        }
    }

}
