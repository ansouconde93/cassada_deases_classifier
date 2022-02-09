package manioc.services;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.JsonProcessingException;

import manioc.classifieur.ConstanteParametrageModele;
import manioc.classifieur.DataNormalizer;
import manioc.classifieur.ModeleComputation;
@Transactional
@Service

public class ManiocMaladieClassifieurServiceImp implements ManiocMaladieClassifieurService{

	@Override
	public void predictImageClass(MultipartFile file) throws IOException {
		// TODO Auto-generated method stub
	       Files.write(Paths.get(new ConstanteParametrageModele().imagesToPredictClasses+"/"+file.getName()+".jpg"),file.getBytes());
	       new DataNormalizer().redimensionneImage(new ConstanteParametrageModele().imagesToPredictClasses);
		
	}

	@Override
	public INDArray predictImageClasse() throws IOException, InterruptedException {
		ModeleComputation computation = new ModeleComputation();
		return computation.prediction();
	}

	@Override
	public Map<String, String> getLabels() throws JsonProcessingException {
		// TODO Auto-generated method stub
		return new DataNormalizer().readOutputFromJSONFile(new ConstanteParametrageModele().outPut);
	}

}
