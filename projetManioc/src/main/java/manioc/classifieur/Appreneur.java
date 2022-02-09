package manioc.classifieur;

import java.io.IOException;

public class Appreneur {

	public static void main(String[] args) throws IOException, InterruptedException {
        new DataNormalizer().preprocessingData();
        new ModeleComputation().testerModele();
	}

}
