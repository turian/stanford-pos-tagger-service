//MaxentTagger -- StanfordMaxEnt, A Maximum Entropy Toolkit
//Copyright (c) 2002-2010 Leland Stanford Junior University


//This program is free software; you can redistribute it and/or
//modify it under the terms of the GNU General Public License
//as published by the Free Software Foundation; either version 2
//of the License, or (at your option) any later version.

//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with this program; if not, write to the Free Software
//Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

//For more information, bug reports, fixes, contact:
//Christopher Manning
//Dept of Computer Science, Gates 1A
//Stanford CA 94305-9010
//USA
//Support/Questions: java-nlp-user@lists.stanford.edu
//Licensing: java-nlp-support@lists.stanford.edu
//http://www-nlp.stanford.edu/software/tagger.shtml


package edu.stanford.nlp.tagger.maxent;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.InDataStreamFile;
import edu.stanford.nlp.io.OutDataStreamFile;
import edu.stanford.nlp.io.PrintFile;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.SentenceProcessor;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.maxent.CGRunner;
import edu.stanford.nlp.maxent.Problem;
import edu.stanford.nlp.maxent.iis.LambdaSolve;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.objectbank.ReaderIteratorFactory;
import edu.stanford.nlp.objectbank.TokenizerFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.ListProcessor;
import edu.stanford.nlp.process.TransformXML;
import edu.stanford.nlp.process.WhitespaceTokenizer;
import edu.stanford.nlp.process.WordToSentenceProcessor;
import edu.stanford.nlp.process.PTBTokenizer.PTBTokenizerFactory;
import edu.stanford.nlp.sequences.PlainTextDocumentReaderAndWriter;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Timing;
import edu.stanford.nlp.util.XMLUtils;


/**
 * The main class for users to run, train, and test the part of speech tagger.
 *
 * You can tag things through the Java API or from the command line.
 * The two English taggers included in this distribution are:
 * <ul>
 * <li> A bi-directional dependency network tagger in models/bidirectional-distsim-wsj-0-18.tagger.
 *      Its accuracy was 97.32% on Penn Treebank WSJ secs. 22-24.</li>
 * <li> A model using only left sequence information and similar but less
 *      unknown words and lexical features as the previous model in
 *      models/left3words-wsj-0-18.tagger. This tagger runs a lot faster.
 *      Its accuracy was 96.92% on Penn Treebank WSJ secs. 22-24.</li>
 * </ul>
 *
 * <h3>Using the Java API</h3>
 * <dl>
 * <dt>
 * A MaxentTagger can be made with a constructor taking as argument the location of parameter files for a trained tagger: </dt>
 * <dd> <code>MaxentTagger tagger = new MaxentTagger("models/left3words-wsj-0-18.tagger");</code></dd>
 * <p>
 * <dt>A default path is provided for the location of the tagger on the Stanford NLP machines:</dt>
 * <dd><code>MaxentTagger tagger = new MaxentTagger(DEFAULT_NLP_GROUP_MODEL_PATH); </code></dd>
 * <p>
 * <dt>If you set the NLP_TAGGER_HOME environment variable,
 * DEFAULT_NLP_GROUP_MODEL_PATH will instead point to the directory
 * given in NLP_TAGGER_HOME.</dt>
 * <p>
 * <dt>To tag a Sentence and get a TaggedSentence: </dt>
 * <dd><code>Sentence taggedSentence = tagger.tagSentence(Sentence sentence)</code></dd>
 * <dd><code>Sentence taggedSentence = tagger.apply(Sentence sentence)</code></dd>
 * <p>
 * <dt>To tag a list of sentences and get back a list of tagged sentences:
 * <dd><code> List taggedList = tagger.process(List sentences)</code></dd>
 * <p>
 * <dt>To tag a String of text and to get back a String with tagged words:</dt>
 * <dd> <code>String taggedString = tagger.tagString("Here's a tagged string.")</code></dd>
 * <p>
 * <dt>To tag a string of <i>correctly tokenized</i>, whitespace-separated words and get a string of tagged words back:</dt>
 * <dd> <code>String taggedString = tagger.tagTokenizedString("Here 's a tagged string .")</code></dd>
 * </dl>
 * <p>
 * The <code>tagString</code> method uses the default tokenizer (PTBTokenizer).
 * If you wish to control tokenization, you may wish to call
 * {@link #tokenizeText(Reader, TokenizerFactory)} and then to call
 * <code>process()</code> on the result.
 * </p>
 *
 * <h3>Using the command line</h3>
 *
 * Tagging, testing, and training can all also be done via the command line.
 * <h3>Training from the command line</h3>
 * To train a model from the command line, first generate a property file:
 * <pre>java edu.stanford.nlp.tagger.maxent.MaxentTagger -genprops </pre>
 *
 * This gets you a default properties file with descriptions of each parameter you can set in
 * your trained model.  You can modify the properties file , or use the default options.  To train, run:
 * <pre>java -mx1g edu.stanford.nlp.tagger.maxent.MaxentTagger -props myPropertiesFile.props </pre>
 *
 *  with the appropriate properties file specified; any argument you give in the properties file can also
 *  be specified on the command line.  You must have specified a model using -model, either in the properties file
 *  or on the command line, as well as a file containing tagged words using -trainFile.
 *
 * Useful flags for controlling the amount of output are -verbose, which prints extra debugging information,
 * and -verboseResults, which prints full information about intermediate results.  -verbose defaults to false
 * and -verboseResults defaults to true.
 *
 * <h3>Tagging and Testing from the command line</h3>
 *
 * Usage:
 * For tagging (plain text):
 * <pre>java edu.stanford.nlp.tagger.maxent.MaxentTagger -model &lt;modelFile&gt; -textFile &lt;textfile&gt; </pre>
 * For testing (evaluating against tagged text):
 * <pre>java edu.stanford.nlp.tagger.maxent.MaxentTagger -model &lt;modelFile&gt; -testFile &lt;testfile&gt; </pre>
 * You can use the same properties file as for training
 * if you pass it in with the "-props" argument. The most important
 * arguments for tagging (besides "model" and "file") are "tokenize"
 * and "tokenizerFactory". See below for more details.
 *
 * Note that the tagger assumes input has not yet been tokenized and by default tokenizes it using a default
 * English tokenizer.  If your input has already been tokenized, use the flag "-tokenized".
 *
 * <p> Parameters can be defined using a Properties file
 * (specified on the command-line with <code>-prop</code> <i>propFile</i>),
 * or directly on the command line (by preceding their name with a minus sign
 * ("-") to turn them into a flag. The following properties are recognized:
 * </p>
 * <table border="1">
 * <tr><td><b>Property Name</b></td><td><b>Type</b></td><td><b>Default Value</b></td><td><b>Relevant Phase(s)</b></td><td><b>Description</b></td></tr>
 * <tr><td>model</td><td>String</td><td>N/A</td><td>All</td><td>Path and filename where you would like to save the model (training) or where the model should be loaded from (testing, tagging).</td></tr>
 * <tr><td>trainFile</td><td>String</td><td>N/A</td><td>Train</td><td>Path to the file holding the training data; specifying this option puts the tagger in training mode.  Only one of 'trainFile','testFile','texFile', and 'convertToSingleFile' may be specified.</td></tr>
 * <tr><td>testFile</td><td>String</td><td>N/A</td><td>Test</td><td>Path to the file holding the test data; specifying this option puts the tagger in testing mode.  Only one of 'trainFile','testFile','texFile', and 'convertToSingleFile' may be specified.</td></tr>
 * <tr><td>textFile</td><td>String</td><td>N/A</td><td>Tag</td><td>Path to the file holding the text to tag; specifying this option puts the tagger in tagging mode.  Only one of 'trainFile','testFile','textFile', and 'convertToSingleFile' may be specified.</td></tr>
 * <tr><td>convertToSingleFile</td><td>String</td><td>N/A</td><td>N/A</td><td>Provided only for backwards compatibility, this option allows you to convert a tagger trained using a previous version of the tagger to the new single-file format.  The value of this flag should be the path for the new model file, 'model' should be the path prefix to the old tagger (up to but not including the ".holder"), and you should supply the properties configuration for the old tagger with -props (before these two arguments).</td></tr>
 * <tr><td>genprops</td><td>boolean</td><td>N/A</td><td>N/A</td><td>Use this option to output a default properties file, containing information about each of the possible configuration options.</td></tr>
 * <tr><td>delimiter</td><td>char</td><td>/</td><td>All</td><td>Delimiter character that separates word and part of speech tags.  For training and testing, this is the delimiter used in the train/test files.  For tagging, this is the character that will be inserted between words and tags in the output.</td></tr>
 * <tr><td>encoding</td><td>String</td><td>UTF-8</td><td>All</td><td>Encoding of the read files (training, testing) and the output text files.</td></tr>
 * <tr><td>tokenize</td><td>boolean</td><td>true</td><td>Tag,Test</td><td>Whether or not the file has been tokenized.  If this is true, the tagger assumes that white space separates all and only those things that should be tagged as separate tokens, and that the input is strictly one sentence per line.</td></tr>
 * <tr><td>tokenizerFactory</td><td>String</td><td>edu.stanford.nlp.process.PTBTokenizer</td><td>Tag,Test</td><td>Fully qualified class name of the tokenizer to use.  edu.stanford.nlp.process.PTBTokenizer does basic English tokenization.</td></tr>
 * <tr><td>tokenizerOptions</td><td>String</td><td></td><td>Tag,Test</td><td>Known options for the particular tokenizer used. A comma-separated list. For PTBTokenizer, options of interest include <code>americanize=false</code> and <code>asciiQuotes</code> (for German). Note that any choice of tokenizer options that conflicts with the tokenization used in the tagger training data will likely degrade tagger performance.</td></tr>
 * <tr><td>arch</td><td>String</td><td>generic</td><td>Train</td><td>Architecture of the model, as a comma-separated list of options, some with a parenthesized integer argument written k here: this determines what features are sed to build your model.  Options are 'left3words', 'left5words', 'bidirectional', 'bidirectional5words', generic', 'sighan2005' (Chinese), 'german', 'words(k),' 'naacl2003unknowns', 'naacl2003conjunctions', wordshapes(k), motleyUnknown, suffix(k), prefix(k), prefixsuffix(k), capitalizationsuffix(k), distsim(s), chinesedictionaryfeatures(s), lctagfeatures, unicodeshapes(k). The left3words architectures are faster, but slightly less accurate, than the bidirectional architectures.  'naacl2003unknowns' was our traditional set of unknown word features, but you can now specify features more flexibility via the various other supported keywords. The 'shapes' options map words to equivalence classes, which slightly increase accuracy.</td></tr>
 * <tr><td>lang</td><td>String</td><td>english</td><td>Train</td><td>Language from which the part of speech tags are drawn. This option determines which tags are considered closed-class (only fixed set of words can be tagged with a closed-class tag, such as prepositions). Defined languages are 'english' (Penn tagset), 'polish' (very rudimentary), 'chinese', 'arabic', 'german', and 'medline'.  </td></tr>
 * <tr><td>openClassTags</td><td>String</td><td>N/A</td><td>Train</td><td>Space separated list of tags that should be considered open-class.  All tags encountered that are not in this list are considered closed-class.  E.g. format: "NN VB"</td></tr>
 * <tr><td>closedClassTags</td><td>String</td><td>N/A</td><td>Train</td><td>Space separated list of tags that should be considered closed-class.  All tags encountered that are not in this list are considered open-class.</td></tr>
 * <tr><td>learnClosedClassTags</td><td>boolean</td><td>false</td><td>Train</td><td>If true, induce which tags are closed-class by counting as closed-class tags all those tags which have fewer unique word tokens than closedClassTagThreshold. </td></tr>
 * <tr><td>closedClassTagThreshold</td><td>int</td><td>int</td><td>Train</td><td>Number of unique word tokens that a tag may have and still be considered closed-class; relevant only if learnClosedClassTags is true.</td></tr>
 * <tr><td>sgml</td><td>boolean</td><td>false</td><td>Tag, Test</td><td>Very basic tagging of the contents of all sgml fields; for more complex mark-up, consider using the xmlInput option.</td></tr>
 * <tr><td>xmlInput</td><td>String</td><td></td><td>Tag, Test</td><td>Give a space separated list of tags in an XML file whose content you would like tagged.  Any internal tags that appear in the content of fields you would like tagged will be discarded; the rest of the XML will be preserved and the original text of specified fields will be replaced with the tagged text.</td></tr>
 * <tr><td>xmlOutput</td><td>String</td><td>""</td><td>Tag</td><td>If a path is given, the tagged data be written out to the given file in xml.  If non-empty, each word will be written out within a word tag, with the part of speech as an attribute.  If original input was XML, this will just appear in the field where the text originally came from.  Otherwise, word tags will be surrounded by sentence tags as well.  E.g., &lt;sentence id="0"&gt;&lt;word id="0" pos="NN"&gt;computer&lt;/word&gt;&lt;/sentence&gt;</td></tr>
 * <tr><td>tagInside</td><td>String</td><td>""</td><td>Tag</td><td>Tags inside elements that match the regular expression given in the String.</td></tr>
 * <tr><td>search</td><td>String</td><td>cg</td><td>Train</td><td>Specify the search method to be used in the optimization method for training.  Options are 'cg' (conjugate gradient) or 'iis' (improved iterative scaling).</td></tr>
 * <tr><td>sigmaSquared</td><td>double</td><td>0.5</td><td>Train</td><td>Sigma-squared smoothing/regularization parameter to be used for conjugate gradient search.  Default usually works reasonably well.</td></tr>
 * <tr><td>iterations</td><td>int</td><td>100</td><td>Train</td><td>Number of iterations to be used for improved iterative scaling.</td></tr>
 * <tr><td>rareWordThresh</td><td>int</td><td>5</td><td>Train</td><td>Words that appear fewer than this number of times during training are considered rare words and use extra rare word features.</td></tr>
 * <tr><td>minFeatureThreshold</td><td>int</td><td>5</td><td>Train</td><td>Features whose history appears fewer than this number of times are discarded.</td></tr>
 * <tr><td>curWordMinFeatureThreshold</td><td>int</td><td>2</td><td>Train</td><td>Words that occur more than this number of times will generate features with all of the tags they've been seen with.</td></tr>
 * <tr><td>rareWordMinFeatureThresh</td><td>int</td><td>10</td><td>Train</td><td>Features of rare words whose histories occur fewer than this number of times are discarded.</td></tr>
 * <tr><td>veryCommonWordThresh</td><td>int</td><td>250</td><td>Train</td><td>Words that occur more than this number of times form an equivalence class by themselves.  Ignored unless you are using ambiguity classes.</td></tr>
 * <tr><td>debug</td><td>boolean</td><td>boolean</td><td>All</td><td>Whether to write debugging information (words, top words, unknown words).  Useful for error analysis.</td></tr>
 * <tr><td>debugPrefix</td><td>String</td><td>N/A</td><td>All</td><td>File (path) prefix for where to write out the debugging information (relevant only if debug=true).</td></tr>
 * </table>
 * <p/>
 *
 * @author Kristina Toutanova
 * @author Miler Lee
 * @author Joseph Smarr
 * @author Anna Rafferty
 * @author Michel Galley
 * @author Christopher Manning
 * @author John Bauer
 */
public class MaxentTagger implements Function<ArrayList<? extends HasWord>,ArrayList<TaggedWord>>, SentenceProcessor, ListProcessor<ArrayList<? extends HasWord>,ArrayList<TaggedWord>> {

  // TODO: Add a flag to lemmatize words (Morphology class) on output of tagging
  // TODO: Make it possible in API for caller to get String output in XML (or TSV) format, rather than only slashTags

  /**
   * The directory from which to get taggers when using
   * DEFAULT_NLP_GROUP_MODEL_PATH.  Normally set to the location of
   * the latest left3words tagger on the NLP machines, but can be
   * changed by setting the environment variable TAGGER_HOME.
   */
  public static final String TAGGER_HOME = ((System.getenv("NLP_TAGGER_HOME") != null) ?
                                            System.getenv("NLP_TAGGER_HOME") :
                                            "/u/nlp/data/pos-tagger/wsj3t0-18-left3words");
  public static final String DEFAULT_NLP_GROUP_MODEL_PATH = new File(TAGGER_HOME, "left3words-wsj-0-18.tagger").getPath();
  public static final String DEFAULT_DISTRIBUTION_PATH = "models/left3words-wsj-0-18.tagger";


  public MaxentTagger() {
  }

  /**
   * Constructor for a tagger using a model stored in a particular file.
   * The <code>modelFile</code> is a filename for the model data.
   * The tagger data is loaded when the
   * constructor is called (this can be slow).
   * This constructor first constructs a TaggerConfig object, which loads
   * the tagger options from the modelFile.
   *
   * @param modelFile filename of the trained model
   * @throws IOException if IO problem
   * @throws ClassNotFoundException when there are errors loading a tagger
   */
  public MaxentTagger(String modelFile) throws IOException, ClassNotFoundException {
    this(modelFile, new TaggerConfig("-model", modelFile), true);
  }

  /**
   * Constructor for a tagger using a model stored in a particular file,
   * with options taken from the supplied TaggerConfig.
   * The <code>modelFile</code> is a filename for the model data.
   * The tagger data is loaded when the
   * constructor is called (this can be slow).
   * This version assumes that the tagger options in the modelFile have
   * already been loaded into the TaggerConfig (if that is desired).
   *
   * @param modelFile filename of the trained model
   * @param config The configuration for the tagger
   * @throws IOException if IO problem
   * @throws ClassNotFoundException when there are errors loading a tagger
   */
  public MaxentTagger(String modelFile, TaggerConfig config)
    throws IOException, ClassNotFoundException
  {
    this(modelFile, config, true);
  }

  /**
   * Initializer that loads the tagger.
   *
   * @param modelFile Where to initialize the tagger from.
   *        Most commonly, this is the filename of the trained model, for example, <code>
   *        /u/nlp/data/pos-tagger/wsj3t0-18-left3words/left3words-wsj-0-18.tagger
   *        </code>.  However, if it starts with "https?://" it will be
   *        interpreted as a URL, and if it starts with "jar:" it will be
   *        taken as a resources in the /models/ path of the current jar file.
   * @param config TaggerConfig based on command-line arguments
   * @param printLoading Whether to print a message saying what model file is being loaded and how long it took when finished.
   * @throws IOException if IO problem
   * @throws ClassNotFoundException when there are errors loading a tagger
   */
  public MaxentTagger(String modelFile, TaggerConfig config, boolean printLoading)
    throws IOException, ClassNotFoundException
  {
    readModelAndInit(config, modelFile, printLoading);
  }


  final Dictionary dict = new Dictionary();
  TTags tags;

  byte[][] fnumArr;
  LambdaSolveTagger prob;
  HashMap<FeatureKey,Integer> fAssociations = new HashMap<FeatureKey,Integer>();
  //PairsHolder pairs = new PairsHolder();
  Extractors extractors;
  Extractors extractorsRare;
  AmbiguityClasses ambClasses;
  final boolean alltags = false;
  final HashMap<String, HashSet<String>> tagTokens = new HashMap<String, HashSet<String>>();

  static final int RARE_WORD_THRESH = 5;
  static final int MIN_FEATURE_THRESH = 5;
  static final int CUR_WORD_MIN_FEATURE_THRESH = 2;
  static final int RARE_WORD_MIN_FEATURE_THRESH = 10;
  static final int VERY_COMMON_WORD_THRESH = 250;

  static final boolean OCCURRING_TAGS_ONLY = false;
  static final boolean POSSIBLE_TAGS_ONLY = false;

  double defaultScore;

  int leftContext;
  int rightContext;

  TaggerConfig config;

  /**
   * Determines which words are considered rare.  All words with count
   * in the training data strictly less than this number (standardly, &lt; 5) are
   * considered rare.
   */
  private int rareWordThresh = RARE_WORD_THRESH;

  /**
   * Determines which features are included in the model.  The model
   * includes features that occurred strictly more times than this number
   * (standardly, &gt; 5) in the training data.  Here I look only at the
   * history (not the tag), so the history appearing this often is enough.
   */
  int minFeatureThresh = MIN_FEATURE_THRESH;

  /**
   * This is a special threshold for the current word feature.
   * Only words that have occurred strictly &gt; this number of times
   * in total will generate word features with all of their occurring tags.
   * The traditional default was 2.
   */
  int curWordMinFeatureThresh = CUR_WORD_MIN_FEATURE_THRESH;

  /**
   * Determines which rare word features are included in the model.
   * The features for rare words have a strictly higher support than
   * this number are included. Traditional default is 10.
   */
  int rareWordMinFeatureThresh = RARE_WORD_MIN_FEATURE_THRESH;

  /**
   * If using tag equivalence classes on following words, words that occur
   * strictly more than this number of times (in total with any tag)
   * are sufficiently frequent to form an equivalence class
   * by themselves. (Not used unless using equivalence classes.)
   *
   * There are places in the code (ExtractorAmbiguityClass.java, for one)
   * that assume this value is constant over the life of a tagger.
   */
  int veryCommonWordThresh = VERY_COMMON_WORD_THRESH;


  int xSize;
  int ySize;
  boolean occuringTagsOnly = OCCURRING_TAGS_ONLY;
  boolean possibleTagsOnly = POSSIBLE_TAGS_ONLY;

  private boolean initted = false;

  // TODO: presumably this should be tied to the command option -verbose
  static final boolean VERBOSE = false;



  /* Package access - shouldn't be part of public API. */
  LambdaSolve getLambdaSolve() {
    return prob;
  }

  // TODO: make these constructors instead of init methods?
  void init(TaggerConfig config) {
    if (initted) return;  // TODO: why not reinit?

    this.config = config;

    String lang, arch;
    String[] openClassTags, closedClassTags;

    if (config == null) {
      lang = "english";
      arch = "left3words";
      openClassTags = StringUtils.EMPTY_STRING_ARRAY;
      closedClassTags = StringUtils.EMPTY_STRING_ARRAY;
    } else {
      lang = config.getLang();
      arch = config.getArch();
      openClassTags = config.getOpenClassTags();
      closedClassTags = config.getClosedClassTags();

      if (((openClassTags.length > 0) && !lang.equals("")) || ((closedClassTags.length > 0) && !lang.equals("")) || ((closedClassTags.length > 0) && (openClassTags.length > 0))) {
        throw new RuntimeException("At least two of lang (\"" + lang + "\"), openClassTags (length " + openClassTags.length + ": " + Arrays.toString(openClassTags) + ")," +
            "and closedClassTags (length " + closedClassTags.length + ": " + Arrays.toString(closedClassTags) + ") specified---you must choose one!");
      } else if ((openClassTags.length == 0) && lang.equals("") && (closedClassTags.length == 0) && ! config.getLearnClosedClassTags()) {
        System.err.println("warning: no language set, no open-class tags specified, and no closed-class tags specified; assuming ALL tags are open class tags");
      }
    }

    if (openClassTags.length > 0) {
      tags = new TTags();
      tags.setOpenClassTags(openClassTags);
    } else if (closedClassTags.length > 0) {
      tags = new TTags();
      tags.setClosedClassTags(closedClassTags);
    } else {
      tags = new TTags(lang);
    }

    defaultScore = lang.equals("english") ? 1.0 : 0.0;

    if (config != null) {
      rareWordThresh = config.getRareWordThresh();
      minFeatureThresh = config.getMinFeatureThresh();
      curWordMinFeatureThresh = config.getCurWordMinFeatureThresh();
      rareWordMinFeatureThresh = config.getRareWordMinFeatureThresh();
      veryCommonWordThresh = config.getVeryCommonWordThresh();
      occuringTagsOnly = config.occuringTagsOnly();
      possibleTagsOnly = config.possibleTagsOnly();
      // System.err.println("occuringTagsOnly: "+occuringTagsOnly);
      // System.err.println("possibleTagsOnly: "+possibleTagsOnly);

      if(config.getDefaultScore() >= 0)
        defaultScore = config.getDefaultScore();
    }

    if (config == null || config.getMode() == TaggerConfig.Mode.TRAIN) {
      // initialize the extractors based on the arch variable
      // you only need to do this when training; otherwise they will be
      // restored from the serialized file
      extractors = new Extractors(ExtractorFrames.getExtractorFrames(arch));
      extractorsRare = new Extractors(ExtractorFramesRare.getExtractorFramesRare(arch, tags));

      setExtractorsGlobal();
    }

    ambClasses = new AmbiguityClasses(tags);

    initted = true;
  }


  /**
   * Figures out what tokenizer factory might be described by the
   * config.  If it's described by name in the config, uses reflection
   * to get the factory (which may cause an exception, of course...)
   */
  public TokenizerFactory<? extends HasWord> chooseTokenizerFactory()
    throws ClassNotFoundException,
           NoSuchMethodException, IllegalAccessException,
           java.lang.reflect.InvocationTargetException
  {
    return chooseTokenizerFactory(config.getTokenize(), config.getTokenizerFactory(), config.getTokenizerOptions());
  }

  protected static TokenizerFactory<? extends HasWord> chooseTokenizerFactory(boolean tokenize,
                                                                              String tokenizerFactory,
                                                                              String tokenizerOptions)
    throws ClassNotFoundException,
           NoSuchMethodException, IllegalAccessException,
           java.lang.reflect.InvocationTargetException
  {
    if (tokenize && tokenizerFactory.trim().length() != 0) {
      //return (TokenizerFactory<? extends HasWord>) Class.forName(getTokenizerFactory()).newInstance();
      @SuppressWarnings({"unchecked"})
        Class<TokenizerFactory<? extends HasWord>> clazz = (Class<TokenizerFactory<? extends HasWord>>) Class.forName(tokenizerFactory.trim());
      Method factoryMethod = clazz.getMethod("newTokenizerFactory");
      @SuppressWarnings({"unchecked"})
        TokenizerFactory<? extends HasWord> factory = (TokenizerFactory<? extends HasWord>) factoryMethod.invoke(tokenizerOptions);
      return factory;
    } else if (tokenize){
      return PTBTokenizerFactory.newWordTokenizerFactory(tokenizerOptions);
    } else {
      return WhitespaceTokenizer.factory();
    }
  }

  /* Package access.  Not part of public API. */
  int getNum(FeatureKey s) {
    Integer num = fAssociations.get(s); // hprof: 15% effective running time
    if (num == null) {
      return -1;
    } else {
      return num;
    }
  }


  // serialize the ExtractorFrames and ExtractorFramesRare in filename
  private void saveExtractors(OutputStream os) throws IOException {

    ObjectOutputStream out = new ObjectOutputStream(os);

    System.out.println(extractors.toString() + "\nrare" + extractorsRare.toString());
    out.writeObject(extractors);
    out.writeObject(extractorsRare);
  }

  // Read the extractors from a filename.
  private void readExtractors(String filename) throws IOException, ClassNotFoundException {
    InputStream in = new BufferedInputStream(new FileInputStream(filename));
    readExtractors(in);
    in.close();
  }

  // Read the extractors from a stream.
  private void readExtractors(InputStream file) throws IOException, ClassNotFoundException {
    ObjectInputStream in = new ObjectInputStream(file);
    extractors = (Extractors) in.readObject();
    extractorsRare = (Extractors) in.readObject();
    extractors.initTypes();
    extractorsRare.initTypes();
    int left = extractors.leftContext();
    int left_u = extractorsRare.leftContext();
    if (left_u > left) {
      left = left_u;
    }
    leftContext = left;
    int right = extractors.rightContext();
    int right_u = extractorsRare.rightContext();
    if (right_u > right) {
      right = right_u;
    }
    rightContext = right;

    setExtractorsGlobal();
  }

  // Sometimes there is data associated with the tagger (such as a
  // dictionary) that we don't want saved with each extractor.  This
  // call lets those extractors get that information from the tagger
  // after being loaded from a data file.
  private void setExtractorsGlobal() {
    extractors.setGlobalHolder(this);
    extractorsRare.setGlobalHolder(this);
  }



  protected void saveModel(String filename, TaggerConfig config) {
    try {
      OutDataStreamFile file = new OutDataStreamFile(filename);
      config.saveConfig(file);
      file.writeInt(xSize);
      file.writeInt(ySize);
      dict.save(file);
      tags.save(file, tagTokens);

      saveExtractors(file);

      file.writeInt(fAssociations.size());
      for (Map.Entry<FeatureKey,Integer> item : fAssociations.entrySet()) {
        int numF = item.getValue();
        file.writeInt(numF);
        FeatureKey fk = item.getKey();
        fk.save(file);
      }

      LambdaSolve.save_lambdas(file, prob.lambda);
      file.close();
    } catch (IOException ioe) {
      System.err.println("Error saving tagger to file " + filename);
      ioe.printStackTrace();
    }
  }


  /**
   * This method is provided for backwards compatibility with the old tagger.  It reads
   * a tagger that was saved as multiple files into the current format and saves it back
   * out as a single file, newFilename.
   *
   * @param filename The name of the holder file, which is also used as a prefix for other filenames
   * @param newFilename The name of the new one-file model that will be written
   * @param config tagger configuration file
   * @return true (whether this operation succeeded; always true
   * @throws ClassNotFoundException especially for incompatible tagger formats
   * @throws IOException if there are errors reading or writing files
   * @throws FileNotFoundException ...
   */
  // TODO: get rid of code duplication here
  private static boolean convertMultifileTagger(String filename, String newFilename, TaggerConfig config) throws ClassNotFoundException, IOException, FileNotFoundException {
    InDataStreamFile rf = new InDataStreamFile(filename);
    MaxentTagger tagger = new MaxentTagger();
    tagger.init(config);
    if (VERBOSE) {
      System.err.println(" length of holder " + new File(filename).length());
    }

    tagger.xSize = rf.readInt();
    tagger.ySize = rf.readInt();
    tagger.dict.read(filename + ".dict");

    if (VERBOSE) {
      System.err.println(" dictionary read ");
    }
    tagger.tags.read(filename + ".tags");
    tagger.readExtractors(filename + ".ex");

    tagger.dict.setAmbClasses(tagger.ambClasses, tagger.veryCommonWordThresh, tagger.tags);

    int[] numFA = new int[tagger.extractors.getSize() +
                          tagger.extractorsRare.getSize()];
    int sizeAssoc = rf.readInt();
    PrintFile pfVP = null;
    if (VERBOSE) {
      pfVP = new PrintFile("pairs.txt");
    }
    for (int i = 0; i < sizeAssoc; i++) {
      int numF = rf.readInt();
      FeatureKey fK = new FeatureKey();
      fK.read(rf);
      numFA[fK.num]++;
      tagger.fAssociations.put(fK, numF);
    }

    if (VERBOSE) {
      pfVP.close();
    }
    if (VERBOSE) {
      for (int k = 0; k < numFA.length; k++) {
        System.err.println(" Number of features of kind " + k + ' ' + numFA[k]);
      }
    }
    tagger.prob = new LambdaSolveTagger(filename + ".prob");
    if (VERBOSE) {
      System.err.println(" prob read ");
    }

    tagger.saveModel(newFilename, config);
    rf.close();
    return true;
  }


  /** This reads the complete tagger from a single model stored in a file, at a URL,
   *  or as a resource
   *  in a jar file, and inits the tagger using a
   *  combination of the properties passed in and parameters from the file.
   *  <p>
   *  <i>Note for the future:</i> This assumes that the TaggerConfig in the file
   *  has already been read and used.  This work is done inside the
   *  constructor of TaggerConfig.  It might be better to refactor
   *  things so that is all done inside this method, but for the moment
   *  it seemed better to leave working code alone [cdm 2008].
   *
   *  @param config The tagger config
   *  @param modelFileOrUrl The name of the model file. This routine opens and closes it.
   *  @param printLoading Whether to print a message saying what model file is being loaded and how long it took when finished.
   *  @throws IOException If I/O errors, etc.
   *  @throws ClassNotFoundException especially for incompatible tagger formats
   */
  protected void readModelAndInit(TaggerConfig config, String modelFileOrUrl, boolean printLoading) throws IOException, ClassNotFoundException {
    // first check can open file ... or else leave with exception
    DataInputStream rf = config.getTaggerDataInputStream(modelFileOrUrl);

    // if (VERBOSE) {
    //   System.err.println(" length of model holder " + new File(modelFileOrUrl).length());
    // }

    readModelAndInit(config, rf, printLoading);
    rf.close();
  }



  /** This reads the complete tagger from a single model file, and inits
   *  the tagger using a combination of the properties passed in and
   *  parameters from the file.
   *  <p>
   *  <i>Note for the future: This assumes that the TaggerConfig in the file
   *  has already been read and used.  It might be better to refactor
   *  things so that is all done inside this method, but for the moment
   *  it seemed better to leave working code alone [cdm 2008].</i>
   *
   *  @param config The tagger config
   *  @param rf DataInputStream to read from.  It's the caller's job to open and close this stream.
   *  @param printLoading Whether to print a message saying what model file is being loaded and how long it took when finished.
   *  @throws IOException If I/O errors
   *  @throws ClassNotFoundException If serialization errors
   */
  protected void readModelAndInit(TaggerConfig config, DataInputStream rf,
                                  boolean printLoading) throws IOException, ClassNotFoundException {
    Timing t = new Timing();
    if (printLoading) t.doing("Reading POS tagger model from " + config.getModel());
    // then init tagger
    init(config);
    TaggerConfig ret = TaggerConfig.readConfig(rf); // taggerconfig in file has already been put into config in constructor of TaggerConfig, so usually just read past it.

    xSize = rf.readInt();
    ySize = rf.readInt();
    dict.read(rf);

    if (VERBOSE) {
      System.err.println(" dictionary read ");
    }
    tags.read(rf);
    readExtractors(rf);
    dict.setAmbClasses(ambClasses, veryCommonWordThresh, tags);

    int[] numFA = new int[extractors.getSize() + extractorsRare.getSize()];
    int sizeAssoc = rf.readInt();
    // init the Hash at the right size for efficiency (avoid resizing ops)
    // mg2008: sizeAssoc defines the number of keys, whereas specifying
    // sizeAssoc as argument defines an initial size.
    // Unless load factor is >= 1, fAssociations is guaranteed to resize at least once.
    //fAssociations = new HashMap<FeatureKey,Integer>(sizeAssoc);
    fAssociations = new HashMap<FeatureKey,Integer>(sizeAssoc*2);
    if (VERBOSE) System.err.printf("Reading %d feature keys...\n",sizeAssoc);
    PrintFile pfVP = null;
    if (VERBOSE) {
      pfVP = new PrintFile("pairs.txt");
    }
    for (int i = 0; i < sizeAssoc; i++) {
      int numF = rf.readInt();
      FeatureKey fK = new FeatureKey();
      fK.read(rf);
      numFA[fK.num]++;
      fAssociations.put(fK, numF);
    }
    if (VERBOSE) {
      pfVP.close();
    }
    if (VERBOSE) {
      for (int k = 0; k < numFA.length; k++) {
        System.err.println(" Number of features of kind " + k + ' ' + numFA[k]);
      }
    }
    prob = new LambdaSolveTagger(rf);
    if (VERBOSE) {
      System.err.println(" prob read ");
    }
    if (printLoading) t.done();
  }


  protected void dumpModel() {
    assert fAssociations.size() == prob.lambda.length;
    for (Map.Entry<FeatureKey,Integer> fk : fAssociations.entrySet()) {
      System.out.println(fk.getKey() + ": " + prob.lambda[fk.getValue()]);
    }
  }


  /* Package access so it doesn't appear in public API. */
  boolean isRare(String word) {
    return dict.sum(word) < rareWordThresh;
  }

  public TTags getTags() {
    return tags;
  }

  /**
   * Tags the tokenized input string and returns the tagged version.
   * This method requires the input to already be tokenized.
   * The tagger wants input that is whitespace separated tokens, tokenized
   * according to the conventions of the training data. (For instance,
   * for the Penn Treebank, punctuation marks and possessive "'s" should
   * be separated from words.)
   *
   * @param toTag The untagged input String
   * @return The same string with tags inserted in the form word/tag
   */
  public String tagTokenizedString(String toTag) {
    ArrayList<Word> sent = Sentence.toUntaggedList(Arrays.asList(toTag.split("\\s+")));
    TestSentence testSentence = new TestSentence(this);
    testSentence.tagSentence(sent);
    return testSentence.getTaggedNice();
  }


  /**
   * Tags the input string and returns the tagged version.
   * This method tokenizes the input into words in perhaps multiple sentences
   * and then tags those sentences.  The default (PTB English)
   * tokenizer is used.
   *
   * @param toTag The untagged input String
   * @return A String of sentences with tags inserted in the form word/tag
   */
  public String tagString(String toTag) {
    TaggerWrapper tw = new TaggerWrapper(this);
    return tw.apply(toTag);
  }

  /**
   * Expects a sentence and returns a tagged sentence.  The input Sentence items
   *
   *
   * @param in This needs to be a Sentence
   * @return A Sentence of TaggedWord
   */
  public ArrayList<TaggedWord> apply(ArrayList<? extends HasWord> in) {
    TestSentence testSentence = new TestSentence(this);
    return testSentence.tagSentence(in);
  }

  /**
   * Tags the Words in each Sentence in the given List with their
   * grammatical part-of-speech. The returned List contains Sentences
   * consisting of TaggedWords.
   * <p><b>NOTE: </b>The input document must contain sentences as its elements,
   * not words. To turn a Document of words into a Document of sentences, run
   * it through {@link WordToSentenceProcessor}.
   *
   * @param sentences A List of Sentence
   * @return A List of Sentence of TaggedWord (final generification cannot be listed due to lack of complete generification of super classes)
   */
  public List<ArrayList<TaggedWord>> process(List<? extends ArrayList<? extends HasWord>> sentences) {
    List<ArrayList<TaggedWord>> taggedSentences = new ArrayList<ArrayList<TaggedWord>>();

    TestSentence testSentence = new TestSentence(this);
    for (ArrayList<? extends HasWord> sentence : sentences) {
      taggedSentences.add(testSentence.tagSentence(sentence));
    }
    return taggedSentences;
  }


  /**
   * Returns a new Sentence that is a copy of the given sentence with all the
   * words tagged with their part-of-speech. Convenience method when you only
   * want to tag a single Sentence instead of a Document of sentences.
   */
  // TODO: genericize... eg genericize the interface we implemented
  @SuppressWarnings({"unchecked"})
  public ArrayList<TaggedWord> processSentence(ArrayList sentence) {
    return tagSentence(sentence);
  }


  /**
   * Returns a new Sentence that is a copy of the given sentence with all the
   * words tagged with their part-of-speech. Convenience method when you only
   * want to tag a single Sentence instead of a Document of sentences.
   * @param sentence sentence to tag
   * @return tagged sentence
   */
  public ArrayList<TaggedWord> tagSentence(List<? extends HasWord> sentence) {
    TestSentence testSentence = new TestSentence(this);
    return testSentence.tagSentence(sentence);
  }

  /**
   * Reads data from r, tokenizes it with the default (Penn Treebank)
   * tokenizer, and returns a List of Sentence objects, which can
   * then be fed into tagSentence.
   *
   * @param r Reader where untokenized text is read
   * @return List of tokenized sentences
   */
  public static List<ArrayList<? extends HasWord>> tokenizeText(Reader r) {
    return tokenizeText(r, null);
  }


  /**
   * Reads data from r, tokenizes it with the given tokenizer, and
   * returns a List of Lists of (extends) HasWord objects, which can then be
   * fed into tagSentence.
   *
   * @param r Reader where untokenized text is read
   * @param tokenizerFactory Tokenizer.  This can be <code>null</code> in which case
   *     the default English tokenizer (PTBTokenizerFactory) is used.
   * @return List of tokenized sentences
   */
  protected static List<ArrayList<? extends HasWord>> tokenizeText(Reader r, TokenizerFactory<? extends HasWord> tokenizerFactory) {
    DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(r);
    if (tokenizerFactory != null) {
      documentPreprocessor.setTokenizerFactory(tokenizerFactory);
    }
    List<ArrayList<? extends HasWord>> out =
      new ArrayList<ArrayList<? extends HasWord>>();
    for (List<HasWord> item : documentPreprocessor) {
      out.add(new ArrayList<HasWord>(item));
    }
    return out;
  }


  /**
   * This method reads in a file in the old multi-file format and saves it to a single file
   * named newName.  The resulting model can then be used with the current architecture. A
   * model must be specified in config that corresponds to the model prefix of the existing
   * multi-file tagger. The new file will be saved to the path specified for the property
   * "convertToSingleFile".
   *
   * @param config The processed form of the command-line arguments.
   */
  private static void convertToSingleFileFormat(TaggerConfig config) {
    try {
      config.dump();
      MaxentTagger.convertMultifileTagger(config.getModel() + ".holder", config.getFile(), config);
    } catch (Exception e) {
      System.err.println("An error occurred while converting to the new tagger format.");
      e.printStackTrace();
    }

  }

  private static void dumpModel(TaggerConfig config) {
    try {
      MaxentTagger tagger = new MaxentTagger(config.getFile(), config, false);
      System.err.println("Serialized tagger built with config:");
      config.dump();
      tagger.dumpModel();
    } catch (Exception e) {
      e.printStackTrace();
    }
  }


  /**
   * Tests a tagger on data with gold tags available.  This is TEST mode.
   *
   * @param config Properties giving parameters for the testing run
   */
  private static void runTest(TaggerConfig config) {
    if (config.getVerbose()) {
      System.err.println("## tagger testing invoked at " + new Date() + " with arguments:");
      config.dump();
    }

    try {
      MaxentTagger tagger = new MaxentTagger(config.getModel(), config);

      Timing t = new Timing();
      TestClassifier testClassifier = new TestClassifier(config, tagger);
      long millis = t.stop();
      printErrWordsPerSec(millis, testClassifier.getNumWords());
      testClassifier.printModelAndAccuracy(config, tagger);
    } catch (Exception e) {
      System.err.println("An error occurred while testing the tagger.");
      e.printStackTrace();
    }
  }


  /**
   * Reads in the training corpus from a filename and trains the tagger
   *
   * @param config Configuration parameters for training a model (filename, etc.
   * @throws IOException If IO problem
   */
  private static void trainAndSaveModel(TaggerConfig config) throws IOException {

    String modelName = config.getModel();
    MaxentTagger maxentTagger = new MaxentTagger();
    maxentTagger.init(config);

    // Allow clobbering.  You want it all the time when running experiments.

    TaggerExperiments samples = new TaggerExperiments(config, maxentTagger);
    TaggerFeatures feats = samples.getTaggerFeatures();
    System.err.println("Samples from " + config.getFile());
    System.err.println("Number of features: " + feats.size());
    Problem p = new Problem(samples, feats);
    LambdaSolveTagger prob = new LambdaSolveTagger(p, 0.0001, 0.00001, maxentTagger.fnumArr);
    maxentTagger.prob = prob;

    if (config.getSearch().equals("owlqn")) {
      CGRunner runner = new CGRunner(prob, config.getModel(), config.getSigmaSquared());
      runner.solveL1(config.getRegL1());
    } else if (config.getSearch().equals("cg")) {
      CGRunner runner = new CGRunner(prob, config.getModel(), config.getSigmaSquared());
      runner.solveCG();
    } else if (config.getSearch().equals("qn")) {
      CGRunner runner = new CGRunner(prob, config.getModel(), config.getSigmaSquared());
      runner.solveQN();
    } else {
      prob.improvedIterative(config.getIterations());
    }

    if (prob.checkCorrectness()) {
      System.out.println("Model is correct [empirical expec = model expec]");
    } else {
      System.out.println("Model is not correct");
    }
    maxentTagger.saveModel(modelName, config);
  }


  /**
   * Trains a tagger model.
   *
   * @param config Properties giving parameters for the training run
   */
  private static void runTraining(TaggerConfig config) {
    Date now = new Date();

    System.err.println("## tagger training invoked at " + now + " with arguments:");
    config.dump();
    Timing tim = new Timing();
    try {
      PrintFile log = new PrintFile(config.getModel() + ".props");
      log.println("## tagger training invoked at " + now + " with arguments:");
      config.dump(log);
      log.close();

      trainAndSaveModel(config);
      tim.done("Training POS tagger");
    } catch(Exception e) {
      System.err.println("An error occurred while training a new tagger.");
      e.printStackTrace();
    }
  }


  public static void printErrWordsPerSec(long milliSec, int numWords) {
    double wordspersec = numWords / (((double) milliSec) / 1000);
    NumberFormat nf = new DecimalFormat("0.00");
    System.err.println("Tagged " + numWords + " words at " +
        nf.format(wordspersec) + " words per second.");
  }


  // not so much a wrapper as a class with some various functionality
  // extending the MaxentTagger...
  // TODO: can we get rid of this? [cdm: sure. I'm not quite sure why Anna added it.  It seems like it could just be inside MaxentTagger]
  static class TaggerWrapper implements Function<String, String> {

    private final TaggerConfig config;
    private final MaxentTagger tagger;
    private TokenizerFactory<? extends HasWord> tokenizerFactory;
    private int sentNum; // = 0;

    protected TaggerWrapper(MaxentTagger tagger) {
      this(null, tagger);
    }

    protected TaggerWrapper(TaggerConfig config, MaxentTagger tagger) {
      this.config = config;
      this.tagger = tagger;
      if (config != null) {
        try {
          tokenizerFactory = chooseTokenizerFactory(config.getTokenize(), 
                                                    config.getTokenizerFactory(), 
                                                    config.getTokenizerOptions());
        } catch (Exception e) {
          System.err.println("Error in tokenizer factory instantiation for class: " + config.getTokenizerFactory());
          e.printStackTrace();
          tokenizerFactory = PTBTokenizerFactory.newWordTokenizerFactory(config.getTokenizerOptions());
        }
      } else {
        tokenizerFactory = PTBTokenizerFactory.newWordTokenizerFactory("");
      }
    }

    public String apply(String o) {
      StringBuilder taggedSentence = new StringBuilder();
      TestSentence testSentence = new TestSentence(tagger);
      final String tagSeparator = ((config == null) ?
                                   null : config.getTagSeparator());
      int outputStyle;
      boolean tokenize;
      if (config != null) {
        outputStyle = PlainTextDocumentReaderAndWriter.asIntOutputFormat(config.getOutputFormat());
        tokenize = config.getTokenize();
      } else {
        outputStyle = PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_SLASH_TAGS;
        tokenize = true;
      }
      if (tokenize) {
        Reader r = new StringReader(o);
        List<ArrayList<? extends HasWord>> l = tagger.tokenizeText(r, tokenizerFactory);

        for (ArrayList<? extends HasWord> s : l) {
          ArrayList<TaggedWord> taggedSentenceTok = testSentence.tagSentence(s);
          if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_TSV) {
            taggedSentence.append(getTsvWords(taggedSentenceTok));
          } else if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_XML ||
            outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_INLINE_XML) {
            taggedSentence.append(getXMLWords(taggedSentenceTok, sentNum++));
          } else { // if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_SLASH_TAGS) {
            taggedSentence.append(Sentence.listToString(taggedSentenceTok, false, tagSeparator)).append(' ');
          }
        }
      } else {
        ArrayList<Word> sent = Sentence.toUntaggedList(Arrays.asList(o.split("\\s+")));
        testSentence.tagSentence(sent);

        if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_TSV) {
          taggedSentence.append(getTsvWords(testSentence.getTaggedSentence()));
        } else if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_XML ||
                   outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_INLINE_XML) {
          taggedSentence.append(getXMLWords(testSentence.getTaggedSentence(), sentNum++));
        } else { // if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_SLASH_TAGS) {
          taggedSentence.append(testSentence.getTaggedNice()).append(' ');
        }
      }
      return taggedSentence.toString();
    }

  } // end class TaggerWrapper

  private static String getXMLWords(ArrayList<TaggedWord> s, int sentNum) {
    StringBuilder sb = new StringBuilder();
    sb.append("<sentence id=\"").append(sentNum).append("\">\n");
    for (int i = 0, sz = s.size(); i < sz; i++) {
      String word = s.get(i).word();
      String tag = s.get(i).tag();
      sb.append("  <word wid=\"").append(i).append("\" pos=\"").append(XMLUtils.escapeAttributeXML(tag)).append("\">").append(XMLUtils.escapeElementXML(word)).append("</word>\n");
    }
    sb.append("</sentence>\n");
    return sb.toString();
  }

  public static String getTsvWords(ArrayList<TaggedWord> s) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0, sz = s.size(); i < sz; i++) {
      String word = s.get(i).word();
      String tag = s.get(i).tag();
      sb.append(word).append('\t').append(tag).append('\n');
    }
    sb.append('\n');
    return sb.toString();
  }

  /**
   * Takes a tagged sentence and writes out the xml version.
   *
   * @param w Where to write the output to
   * @param s A tagged sentence
   * @param sentNum The sentence index for XML printout
   */
  private static void writeXMLSentence(Writer w, ArrayList<TaggedWord> s, int sentNum) {
    try {
      w.write(getXMLWords(s, sentNum));
    } catch (IOException e) {
      System.err.println("Error writing sentence " + sentNum + ": " +
                         Sentence.listToString(s));
      throw new RuntimeIOException(e);
    }
  }

  /**
   * Uses an XML transformer to turn an input stream into a bunch of
   * output.  Tags all of the text between xmlTags.
   *
   * The difference between using this and using runTagger in XML mode
   * is that this preserves the XML structure outside of the list of
   * elements to tag, whereas the runTagger method throws away all of
   * the surrounding structure and returns tagged plain text.
   */
  public void tagFromXML(InputStream input, Writer writer, String ... xmlTags) {
    TransformXML<String> txml = new TransformXML<String>();
    txml.transformXML(xmlTags, new TaggerWrapper(config, this),
                      input, writer, new TransformXML.SAXInterface<String>());
  }

  public void tagFromXML(Reader input, Writer writer, String ... xmlTags) {
    TransformXML<String> txml = new TransformXML<String>();
    txml.transformXML(xmlTags, new TaggerWrapper(config, this),
                      input, writer, new TransformXML.SAXInterface<String>());
  }

  private void tagFromXML() {
    InputStream is = null;
    Writer w = null;
    try {
      is = new BufferedInputStream(new FileInputStream(config.getFile()));
      String outFile = config.getOutputFile();
      if (outFile.length() > 0) {
        w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile),
                                                      config.getEncoding()));
      } else {
        w = new PrintWriter(System.out);
      }
      tagFromXML(is, w, config.getXMLInput());
    } catch (FileNotFoundException e) {
      System.err.println("Input file not found: " + config.getFile());
      e.printStackTrace();
    } catch (IOException ioe) {
      System.err.println("tagFromXML: mysterious IO Exception");
      ioe.printStackTrace();
    } finally {
      IOUtils.closeIgnoringExceptions(is);
      IOUtils.closeIgnoringExceptions(w);
    }
  }

  /**
   * Loads the tagger from a config file and then runs it in TAG mode.
   *
   * @param config The configuration parameters for the run.
   */
  @SuppressWarnings({"unchecked", "UnusedDeclaration"})
  private static void runTagger(TaggerConfig config)
    throws IOException, ClassNotFoundException,
           NoSuchMethodException, IllegalAccessException,
           java.lang.reflect.InvocationTargetException
  {
    if (config.getVerbose()) {
      Date now = new Date();
      System.err.println("## tagger invoked at " + now + " with arguments:");
      config.dump();
    }
    MaxentTagger tagger = new MaxentTagger(config.getModel(), config);
    tagger.runTagger();
  }

  /**
   * Runs the tagger when we're in TAG mode.
   * In this mode, the config contains either the name of the file to
   * tag or stdin.  That file or input is then tagged.
   */
  private void runTagger()
    throws IOException, ClassNotFoundException,
           NoSuchMethodException, IllegalAccessException,
           java.lang.reflect.InvocationTargetException
  {
    String[] xmlInput = config.getXMLInput();
    if (xmlInput.length > 0) {
      if(xmlInput.length > 1 || !xmlInput[0].equals("null")) {
        tagFromXML();
        return;
      }
    }

    BufferedWriter writer = null;
    try {
      String outFile = config.getOutputFile();
      if (outFile.length() > 0) {
        writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile), config.getEncoding()));
      } else {
        writer = new BufferedWriter(new OutputStreamWriter(System.out, config.getEncoding()));
      }

      //Now determine if we're tagging from stdin or from a file,
      //construct a reader accordingly
      boolean stdin = config.useStdin();
      BufferedReader br;
      if (!stdin) {
        br = IOUtils.readReaderFromString(config.getFile(), config.getEncoding());
      } else {
        System.err.println("Type some text to tag, then EOF.");
        System.err.println("  (For EOF, use Return, Ctrl-D on Unix; Enter, Ctrl-Z, Enter on Windows.)");
        br = new BufferedReader(new InputStreamReader(System.in));
      }

      runTagger(br, writer, config.getTagInside(), stdin);
    } finally {
      if (writer != null) {
        IOUtils.closeIgnoringExceptions(writer);
      }
    }
  }

  /**
   * This method runs the tagger on the provided reader & writer.
   *
   * It takes into from the given <code>reader</code>, applies the
   * tagger to it one sentence at a time (determined using
   * documentPreprocessor), and writes the output to the given
   * <code>writer</code>.
   *
   * The document is broken into sentences using the sentence
   * processor determined in the tagger's TaggerConfig.
   *
   * <code>tagInside</code> makes the tagger run in XML mode... if set
   * to non-empty, instead of processing the document as one large
   * text blob, it considers each region in between the given tag to
   * be a separate text blob.
   *
   * <code>stdin</code> makes the tagger take lines one at a time from
   * the reader and print out a bit more nice formatting to the
   * writer.  It doesn't actually require the reader to be stdin,
   * though.  This is mutually exclusive with tagInside; if tagInside
   * is not empty, stdin will be ignored.
   */
  public void runTagger(BufferedReader reader, BufferedWriter writer,
                        String tagInside, boolean stdin)
    throws IOException, ClassNotFoundException,
           NoSuchMethodException, IllegalAccessException,
           java.lang.reflect.InvocationTargetException
  {
    Timing t = new Timing();

    final String sentenceDelimiter = config.getSentenceDelimiter();
    final TokenizerFactory<? extends HasWord> tokenizerFactory =
      chooseTokenizerFactory();

    //Counts
    int numWords = 0;
    int numSentences = 0;

    // it seems too complicated and quite unnecessary to allow people
    // to type XML at the input prompt
    if (tagInside != null && !tagInside.equals(""))
      stdin = false;

    do {
      int outputStyle = PlainTextDocumentReaderAndWriter.asIntOutputFormat(config.getOutputFormat());
      if (config.getSGML()) {
        // this uses NER codebase technology to read/write SGML-ish files
        PlainTextDocumentReaderAndWriter readerAndWriter = new PlainTextDocumentReaderAndWriter();
        ObjectBank<List<CoreLabel>> ob = new ObjectBank<List<CoreLabel>>(new ReaderIteratorFactory(reader), readerAndWriter);
        PrintWriter pw = new PrintWriter(writer);
        for (List<CoreLabel> sentence : ob) {
          ArrayList<CoreLabel> s = new ArrayList<CoreLabel>(sentence);
          numWords += s.size();
          ArrayList<TaggedWord> taggedSentence = tagSentence(s);
          Iterator<CoreLabel> origIter = sentence.iterator();
          for (TaggedWord tw : taggedSentence) {
            CoreLabel cl = origIter.next();
            cl.set(CoreAnnotations.AnswerAnnotation.class, tw.tag());
          }
          readerAndWriter.printAnswers(sentence, pw, outputStyle, true);
        }
      } else {
        //Now we do everything through the doc preprocessor
        List<List<? extends HasWord>> document;
        final DocumentPreprocessor docProcessor;
        if (tagInside.length() > 0) {
          docProcessor =
            new DocumentPreprocessor(reader,
                                      DocumentPreprocessor.DocType.XML);
          docProcessor.setElementDelimiter(tagInside);
        } else if (stdin) {
          String line = reader.readLine();
          // this happens when we reach end of file
          if (line == null)
            break;
          docProcessor =
            new DocumentPreprocessor(new BufferedReader(new StringReader(line)));
        } else {
          docProcessor = new DocumentPreprocessor(reader);
          docProcessor.setSentenceDelimiter(sentenceDelimiter);
        }
        docProcessor.setTokenizerFactory(tokenizerFactory);
        docProcessor.setEncoding(config.getEncoding());

        for (List<HasWord> sentence : docProcessor) {
          numWords += sentence.size();
          ArrayList<TaggedWord> taggedSentence = tagSentence(sentence);

          if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_TSV) {
            writer.write(getTsvWords(taggedSentence));
          } else if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_XML) {
            writeXMLSentence(writer, taggedSentence, numSentences);
          } else { // if (outputStyle == PlainTextDocumentReaderAndWriter.OUTPUT_STYLE_SLASH_TAGS) {
            writer.write(Sentence.listToString(taggedSentence, false, config.getTagSeparator()));
            writer.newLine();
          }
          if (stdin) {
            writer.newLine();
            writer.flush();
          }
          numSentences++;
        }
      }
    } while (stdin); // only go through multiple times in stdin mode
    writer.flush();
    long millis = t.stop();
    printErrWordsPerSec(millis, numWords);
  }


  /**
   * Command-line tagger interface.
   * Can be used to train or test taggers, or to tag text, taking input from
   * stdin or a file.
   * See class documentation for usage.
   *
   * @param args Command-line arguments
   * @throws IOException If any file problems
   */
  public static void main(String[] args) throws Exception {
    TaggerConfig config = new TaggerConfig(args);

    if (config.getMode() == TaggerConfig.Mode.TRAIN) {
      runTraining(config);
    } else if (config.getMode() == TaggerConfig.Mode.TAG) {
      runTagger(config);
    } else if (config.getMode() == TaggerConfig.Mode.TEST) {
      runTest(config);
    } else if (config.getMode() == TaggerConfig.Mode.CONVERT) {
      convertToSingleFileFormat(config);
    } else if (config.getMode() == TaggerConfig.Mode.DUMP) {
      dumpModel(config);
    } else {
      System.err.println("Impossible: nothing to do. None of train, tag, test, or convert was specified.");
    }
  } // end main()

}
