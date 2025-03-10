using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;

namespace CollinsTagger
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                if (args.Length == 0)
                {
                    Console.WriteLine("Usage: CollinsTagger [train|tag] [options]");
                    return;
                }

                Options options = ParseArguments(args.Length > 1 ? 
                    args.Skip(1).ToArray() : Array.Empty<string>());

                if ("train".Equals(args[0], StringComparison.OrdinalIgnoreCase))
                {
                    Train(options);
                }
                else if ("tag".Equals(args[0], StringComparison.OrdinalIgnoreCase))
                {
                    Tag(options);
                }
                else
                {
                    Console.WriteLine($"Unknown command: {args[0]}. Use 'train' or 'tag'.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

        static void Train(Options options)
        {
            Dictionary<string, int> tagMap = new Dictionary<string, int>();
            Dictionary<string, int> featureMap = new Dictionary<string, int>();

            using (StreamReader reader = new StreamReader(options.DataFile))
            {
                string line;
                int t = 0;
                int f = 0;
                while ((line = reader.ReadLine()) != null)
                {
                    if (line != null && !string.IsNullOrEmpty(line.Trim()))
                    {
                        var tokens = Regex.Split(line, @"\s+");
                        if (!tagMap.ContainsKey(tokens[0]))
                        {
                            tagMap.Add(tokens[0], t++);
                        }

                        if (!options.UseFeatureHashing)
                        {
                            for (int i = 1; i < tokens.Length; i++)
                            {
                                if (!featureMap.ContainsKey(tokens[i]))
                                {
                                    featureMap.Add(tokens[i], f++);
                                }
                            }
                        }
                    }
                }
            }

            //key variables
            int numTags = tagMap.Count;
            int numFeatures = options.UseFeatureHashing ? (1 << options.HashBits) : featureMap.Count;

            Trainer trainer = new Trainer(numTags, numFeatures);
            int c = 0;

            //use binary cache for multiple passes
            string cacheFile = Path.Combine(options.BasePath, ".cache");
            IFormatter binaryFormatter = new BinaryFormatter();
            using (Stream writer = new GZipStream(new FileStream(cacheFile, FileMode.Create, FileAccess.Write), CompressionMode.Compress))
            {
                foreach (var instance in ReadInstances(tagMap, featureMap, options))
                {
                    trainer.LearnFromOneInstance(instance.WordsWithFeatures, instance.LabelledTags);
                    c++;

                    if (options.NumIterations > 1)
                    {
                        binaryFormatter.Serialize(writer, instance);
                    }
                }
            }

            Instance cachedInstance = null;
            for (int i = 1; i < options.NumIterations; i++)
            {
                using (Stream reader = new FileStream(cacheFile, FileMode.Open, FileAccess.Read))
                using (GZipStream compressedReader = new GZipStream(reader, CompressionMode.Decompress))
                { 
                    try
                    {
                        while (reader.Position < reader.Length && (cachedInstance = binaryFormatter.Deserialize(compressedReader) as Instance) != null)
                        {
                            trainer.LearnFromOneInstance(cachedInstance.WordsWithFeatures, cachedInstance.LabelledTags);
                            c++;
                        }
                    }
                    catch (SerializationException ex)
                    {
                        Console.WriteLine($"Error deserializing data: {ex.Message}");
                        break;
                    }
                }
            }

            //save model
            Model model = trainer.GetModel();
            string TagDictionary = Path.Combine(options.BasePath, "TagDictionary.txt");
            using (StreamWriter writer = new StreamWriter(TagDictionary))
            {
                foreach (var kvp in tagMap)
                    writer.WriteLine("{0}\t{1}", kvp.Key, kvp.Value);
            }

            if (!options.UseFeatureHashing)
            {
                string FeatureDictionary = Path.Combine(options.BasePath, "FeatureDictionary.txt");
                using (StreamWriter writer = new StreamWriter(FeatureDictionary))
                {
                    foreach (var kvp in featureMap)
                        writer.WriteLine("{0}\t{1}", kvp.Key, kvp.Value);
                }
            }

            string TagTransitionProbabilities = Path.Combine(options.BasePath, "TagTransitionProbabilities.txt");
            using (StreamWriter writer = new StreamWriter(TagTransitionProbabilities))
            {
                for (int t1 = 0; t1 < numTags; t1++)
                    for (int t2 = 0; t2 < numTags; t2++)
                        if (model.alphaTagPreviousTag[t1, t2] != 0)
                            writer.WriteLine("{0}\t{1}\t{2}", t1, t2, model.alphaTagPreviousTag[t1, t2]);
            }

            string TagFeatureProbabilities = Path.Combine(options.BasePath, "TagFeatureProbabilities.txt");
            using (StreamWriter writer = new StreamWriter(TagFeatureProbabilities))
            {
                for (int t = 0; t < numTags; t++)
                    for (int f = 0; f < numFeatures; f++)
                        if (model.alphaTagFeature[t, f] != 0)
                            writer.WriteLine("{0}\t{1}\t{2}", t, f, model.alphaTagFeature[t, f]);
            }
        }

        //tag sequences based on model provided
        static void Tag(Options options)
        {
            //read tag dictionary
            Dictionary<string, int> tagMap = new Dictionary<string,int>();
            Dictionary<int, string> rTagMap = new Dictionary<int, string>();

            string TagDictionary = Path.Combine(options.BasePath, "TagDictionary.txt");
            using (StreamReader reader = new StreamReader(TagDictionary))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var tokens = Regex.Split(line, @"\s+");
                    tagMap.Add(tokens[0], int.Parse(tokens[1]));
                    rTagMap.Add(int.Parse(tokens[1]), tokens[0]);
                }
            }

            //read feature dictionary
            Dictionary<string, int> featureMap = new Dictionary<string, int>();
            if (!options.UseFeatureHashing)
            {
                string FeatureDictionary = Path.Combine(options.BasePath, "FeatureDictionary.txt");
                using (StreamReader reader = new StreamReader(FeatureDictionary))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        var tokens = Regex.Split(line, @"\s+");
                        featureMap.Add(tokens[0], int.Parse(tokens[1]));
                    }
                }
            }

            //key variables
            int numTags = tagMap.Count;
            int numFeatures = options.UseFeatureHashing ? (1 << options.HashBits) : featureMap.Count;

            Model model = new Model()
            {
                alphaTagFeature = new float[numTags, numFeatures],
                alphaTagPreviousTag = new float[numTags, numTags]
            };

            //populate model
            string TagTransitionProbabilities = Path.Combine(options.BasePath, "TagTransitionProbabilities.txt");
            using (StreamReader reader = new StreamReader(TagTransitionProbabilities))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var tokens = Regex.Split(line, @"\s+");
                    model.alphaTagPreviousTag[int.Parse(tokens[0]), int.Parse(tokens[1])] = float.Parse(tokens[2]);
                }
            }
            string TagFeatureProbabilities = Path.Combine(options.BasePath, "TagFeatureProbabilities.txt");
            using (StreamReader reader = new StreamReader(TagFeatureProbabilities))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var tokens = Regex.Split(line, @"\s+");
                    model.alphaTagFeature[int.Parse(tokens[0]), int.Parse(tokens[1])] = float.Parse(tokens[2]);
                }
            }

            //mantian data for precision recall reports
            Dictionary<int, int> perTagCorrect = new Dictionary<int, int>();
            Dictionary<int, int> perTagCount = new Dictionary<int, int>();
            Dictionary<int, int> perTagModelCount = new Dictionary<int, int>();
            Enumerable.Range(0, numTags).ToList().ForEach(x =>
            {
                perTagCount[x] = 0;
                perTagCorrect[x] = 0;
                perTagModelCount[x] = 0;
            });

            int instanceCorrectCount = 0;
            int instanceCount = 0;

            Tagger tagger = new Tagger(numTags, model.alphaTagFeature, model.alphaTagPreviousTag);
            int[] tags = new int[Viterbi.MAX_WORDS];
            using (StreamWriter writer = new StreamWriter(options.Output))
            {
                foreach (var instance in ReadInstances(tagMap, featureMap, options))
                {
                    tagger.Label(instance.WordsWithFeatures, tags);
                    writer.WriteLine(string.Join(Environment.NewLine, Enumerable
                        .Range(0, instance.WordsWithFeatures.Length)
                        .Select(x => rTagMap[tags[x]])
                        .ToArray()));
                    writer.WriteLine();

                    bool allCorrect = true;
                    for (int i = 0; i < instance.WordsWithFeatures.Length; i++)
                    {
                        int correctTag = instance.LabelledTags[i];
                        int modelTag = tags[i];

                        perTagCount[correctTag]++;
                        perTagModelCount[modelTag]++;
                        if (correctTag == modelTag)
                        {
                            perTagCorrect[correctTag]++;
                        }
                        else
                        {
                            allCorrect = false;
                        }
                    }

                    if (allCorrect)
                        instanceCorrectCount++;
                    instanceCount++;
                }
            }
            for (int i = 0; i < numTags; i++)
            {
                double precision = perTagModelCount[i] > 0 ? perTagCorrect[i] * 1.0 / perTagModelCount[i] : 0;
                double recall = perTagCount[i] > 0 ? perTagCorrect[i] * 1.0 / perTagCount[i] : 0;
                
                Console.WriteLine(string.Join("\t", new object[] {
                        rTagMap[i],
                        perTagModelCount[i],
                        perTagCorrect[i],
                        perTagCount[i],
                        precision,
                        recall
                    }.Select(x => x.ToString()).ToArray()));
            }
            Console.WriteLine(string.Join("\t", new object[] {
                instanceCorrectCount,
                instanceCount,
                instanceCorrectCount * 1.0 / instanceCount
            }.Select(x => x.ToString()).ToArray()));
        }

        static Options ParseArguments(string[] args)
        {
            Options options = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                if ("--usefeaturehashing".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    options.UseFeatureHashing = true;
                    if (i + 1 < args.Length)
                    {
                        if (int.TryParse(args[++i], out int hashBits) && hashBits > 0 && hashBits < 31)
                        {
                            options.HashBits = hashBits;
                        }
                        else
                        {
                            throw new ArgumentException($"HashBits must be a positive integer less than 31, got: {args[i]}", "HashBits");
                        }
                    }
                    else
                    {
                        throw new ArgumentException("Missing value for --usefeaturehashing parameter", "args");
                    }
                }
                else if ("--basepath".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < args.Length)
                    {
                        options.BasePath = args[++i];
                    }
                    else
                    {
                        throw new ArgumentException("Missing value for --basepath parameter", "args");
                    }
                }
                else if ("--numiterations".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < args.Length && int.TryParse(args[++i], out int numIter))
                    {
                        options.NumIterations = numIter;
                    }
                    else
                    {
                        throw new ArgumentException("Invalid or missing value for --numiterations parameter", "args");
                    }
                }
                else if ("--data".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < args.Length)
                    {
                        options.DataFile = args[++i];
                        if (!File.Exists(options.DataFile))
                        {
                            throw new FileNotFoundException($"Data file not found: {options.DataFile}");
                        }
                    }
                    else
                    {
                        throw new ArgumentException("Missing value for --data parameter", "args");
                    }
                }
                else if ("--output".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 < args.Length)
                    {
                        options.Output = args[++i];
                    }
                    else
                    {
                        throw new ArgumentException("Missing value for --output parameter", "args");
                    }
                }
                else
                {
                    throw new ArgumentException($"Unknown argument: {args[i]}", "args");
                }
            }
            return options;
        }

        // Read instance from data file
        static IEnumerable<Instance> ReadInstances(
            Dictionary<string,int> tagMap,
            Dictionary<string,int> featureMap,
            Options options)
        {
            using (StreamReader reader = new StreamReader(options.DataFile))
            {
                string line;
                List<int[]> features = new List<int[]>();
                List<int> labelledTags = new List<int>();

                while (!reader.EndOfStream)
                {
                    //read one example
                    while ((line = reader.ReadLine()) != null && !string.IsNullOrEmpty(line))
                    {
                        string[] tokens = Regex.Split(line, @"\s+");

                        //if using feature hashing, dont maintain a feature name to id map
                        //instead use hasing to compute it on the fly
                        if (options.UseFeatureHashing)
                        {
                            features.Add(tokens
                                .Skip(1)
                                .Select(x =>
                                    MurMurHash3.Hash(Encoding.UTF8.GetBytes(x)) & ((1 << options.HashBits) - 1))
                                .Distinct()
                                .ToArray());

                        }
                        else
                        {
                            features.Add(tokens
                                .Skip(1)
                                .Where(x => featureMap.ContainsKey(x))
                                .Select(x => featureMap[x])
                                .Distinct()
                                .ToArray());
                        }

                        //if tag is unseen, try to use "O" tag if it exists, otherwise use the first tag
                        int tagId;
                        if (tagMap.ContainsKey(tokens[0]))
                        {
                            tagId = tagMap[tokens[0]];
                        }
                        else if (tagMap.ContainsKey("O"))
                        {
                            tagId = tagMap["O"];
                        }
                        else if (tagMap.Count > 0)
                        {
                            tagId = tagMap.Values.First();
                        }
                        else
                        {
                            // This should never happen as we already populated tagMap
                            tagId = 0;
                        }
                        labelledTags.Add(tagId);
                    }

                    //skip the sequence if its too long
                    if (features.Count > Viterbi.MAX_WORDS)
                        continue;

                    yield return new Instance()
                    {
                        WordsWithFeatures = features.ToArray(),
                        LabelledTags = labelledTags.ToArray()
                    };

                    features.Clear();
                    labelledTags.Clear();
                }
            }
        }
    }

    [Serializable]
    class Instance
    {
        public int[][] WordsWithFeatures { get; set; }
        public int[] LabelledTags { get; set; }
    }

    class Options
    {
        private string _basePath;
        private int _hashBits;
        private string _dataFile;
        private int _numIterations = 1; // Default to 1 iteration
        private string _output;

        public string BasePath 
        { 
            get => _basePath ?? Environment.CurrentDirectory;
            set => _basePath = value; 
        }
        
        public bool UseFeatureHashing { get; set; }
        
        public int HashBits 
        { 
            get => _hashBits;
            set 
            {
                if (value <= 0 || value >= 31)
                {
                    throw new ArgumentOutOfRangeException(nameof(HashBits), "Hash bits must be between 1 and 30");
                }
                _hashBits = value;
            }
        }
        
        public string DataFile 
        { 
            get => _dataFile;
            set 
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    throw new ArgumentException("Data file path cannot be empty", nameof(DataFile));
                }
                _dataFile = value;
            }
        }
        
        public int NumIterations 
        { 
            get => _numIterations;
            set 
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(NumIterations), "Number of iterations must be positive");
                }
                _numIterations = value;
            }
        }
        
        public string Output 
        { 
            get => _output;
            set 
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    throw new ArgumentException("Output path cannot be empty", nameof(Output));
                }
                _output = value;
            }
        }
    }
}
