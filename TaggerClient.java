
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import org.apache.xmlrpc.XmlRpcClient;


public class TaggerClient {

	public static void main(String[] args) {
		
		int port = 8000;
		
		assert (args.length >= 1): "No input file provided";
		
		try {
			if (args.length == 2 )
				port = Integer.parseInt(args[1]);
		} catch (NumberFormatException e1) {			
			e1.printStackTrace();
		}
		
		try {
			XmlRpcClient client = new XmlRpcClient("127.0.0.1", port);
			Vector<String> v = new Vector<String>();
			System.out.println("Input document: " + args[0]);
			v.add(readFile(args[0]));
			String output = (String) client.execute("tagger.runTagger", v);
			System.out.println("\nmessage received: " + output);
		} catch (Exception e) {
			e.printStackTrace();			
		}
	}

	public static String readFile(String filename) {
		File f = new File(filename);
		StringBuilder contents = new StringBuilder();
		try {
			BufferedReader input = new BufferedReader(new FileReader(f));
			try {
				String line = null; 
				while ((line = input.readLine()) != null) {
					contents.append(line);
				}
			} finally {
				input.close();
			}
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		return contents.toString();
	}
}
