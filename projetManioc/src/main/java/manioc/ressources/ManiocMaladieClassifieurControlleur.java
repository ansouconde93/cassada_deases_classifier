package manioc.ressources;

import java.io.IOException;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import manioc.services.ManiocMaladieClassifieurService;

@RestController
@CrossOrigin("*")
public class ManiocMaladieClassifieurControlleur {
	@Autowired
	ManiocMaladieClassifieurService classifieurService;
	
	 @PostMapping("/uploadManiocImage")
    public void predictImageClass(@RequestParam("file") MultipartFile file) {
    	
    	try {
			classifieurService.predictImageClass(file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
	 @GetMapping("/prediction")
    public INDArray predictImageClasse() {
    	try {
			return classifieurService.predictImageClasse();
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			return null;
		}
    }
	 

	 @GetMapping("/labels")
    public Map<String, String> getLabels() {
    	try {
			return classifieurService.getLabels();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			return null;
		}
    }
}
