package manioc.services;

import java.io.IOException;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.core.JsonProcessingException;

public interface ManiocMaladieClassifieurService {
	public void predictImageClass(MultipartFile file) throws IOException;
	public INDArray predictImageClasse() throws IOException, InterruptedException;
	public Map<String, String> getLabels() throws JsonProcessingException;
}
