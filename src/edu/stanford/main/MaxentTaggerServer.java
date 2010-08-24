
package edu.stanford.main;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.StringReader;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;

import org.apache.xmlrpc.WebServer;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.sequences.PlainTextDocumentReaderAndWriter;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.tagger.maxent.TaggerConfig;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.XMLUtils;


/*
 * @author Ali Afshar (it.zvision@yahoo.com)
 * @version 1.0
 */
 
public class MaxentTaggerServer  {

	private MaxentTagger taggerInstance;	
	private TaggerConfig config; 	
	
	
	public MaxentTaggerServer(TaggerConfig c) throws IOException, ClassNotFoundException {
		config = c;
		taggerInstance = new MaxentTagger(c.getModel(), c);		
	}
	
		
	//Copy from edu/stanford/nlp/tagger/maxent/MaxentTagger.java 
	//public void runTagger(BufferedReader reader, BufferedWriter writer,String tagInside, boolean stdin)
	
	public String runTagger(String input) throws IOException,
			ClassNotFoundException, NoSuchMethodException,
			IllegalAccessException, java.lang.reflect.InvocationTargetException {
				
		String tagInside = config.getTagInside();
		StringReader sr = new StringReader(input);
		BufferedReader reader = new BufferedReader(sr);
		
		Timing t = new Timing();

		final String sentenceDelimiter = config.getSentenceDelimiter();
		final TokenizerFactory<? extends HasWord> tokenizerFactory = taggerInstance.chooseTokenizerFactory(); 

		//Counts
		int numWords = 0;
		int numSentences = 0;
		
		int outputStyle = PlainTextDocumentReaderAndWriter.asIntOutputFormat(config.getOutputFormat());
	
		//Now we do everything through the doc preprocessor
		
		final DocumentPreprocessor docProcessor;
		if (tagInside.length() > 0) {
			docProcessor = new DocumentPreprocessor(reader,
					DocumentPreprocessor.DocType.XML);
			docProcessor.setElementDelimiter(tagInside);
		} else {
			docProcessor = new DocumentPreprocessor(reader);
			docProcessor.setSentenceDelimiter(sentenceDelimiter);
		}
		docProcessor.setTokenizerFactory(tokenizerFactory);
		docProcessor.setEncoding(config.getEncoding());

		StringBuffer buffer = new StringBuffer();

		for (List<HasWord> sentence : docProcessor) {
			numWords += sentence.size();
			ArrayList<TaggedWord> taggedSentence = taggerInstance.tagSentence(sentence);
			
			if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_TSV) {
				String str = MaxentTagger.getTsvWords(taggedSentence);				
				buffer.append(str);

			} else if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_XML) {				
				String str = writeXMLSentence(taggedSentence, numSentences);
				buffer.append(str);
				
			} else { 
				String str = Sentence.listToString(taggedSentence, false,
						config.getTagSeparator());
				buffer.append(str).append(System.getProperty("line.separator"));
			}
			 numSentences++;
		}
		
		long millis = t.stop();
		MaxentTagger.printErrWordsPerSec(millis, numWords);	
		return buffer.toString();
	}
	
	
	private static String writeXMLSentence(ArrayList<TaggedWord> s, int sentNum) {
		StringBuilder sb = new StringBuilder();
		sb.append("<sentence id=\"").append(sentNum).append("\">\n");
		for (int i = 0, sz = s.size(); i < sz; i++) {
			String word = s.get(i).word();
			String tag = s.get(i).tag();
			sb.append("  <word wid=\"").append(i).append("\" pos=\"").append(
					XMLUtils.escapeAttributeXML(tag)).append("\">").append(
					XMLUtils.escapeElementXML(word)).append("</word>\n");
		}
		sb.append("</sentence>\n");
		return sb.toString();
	}
	
	
	
	public static void main(String[] args) throws Exception {
		
		String localhost = "127.0.0.1";	
		try {
			localhost = InetAddress.getLocalHost().getHostAddress();				
		} catch (UnknownHostException e1) {					
			System.err.println("Bad address.\n"+e1.getMessage());		
		}		
		System.out.println("Accepting request from addesses: " + localhost);
		
		//TaggerConfig config = new TaggerConfig("-model", model, "-outputFormat", "xml", "-serverPort", "8090");
		TaggerConfig config = new TaggerConfig(args);
		MaxentTaggerServer tagger = new MaxentTaggerServer(config);
		
		// we have everything we need, start the server
		try {
			WebServer server = new WebServer(config.getServerPort());
			//server.setParanoid(true);
			server.acceptClient(localhost);
			server.addHandler("tagger", tagger);
			server.start();
			System.out.println("MaxentTaggerServer started....Awaiting requests.");
		} catch (Exception e) {
			//java.lang.RuntimeException: Address already in use: JVM_Bind
			System.out.println(e.getMessage());
		}
	}
	
}
