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
            Options options = ParseArguments(Regex.Split(string.Join(" ", args.Skip(1).ToArray()), @"\s+"));

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
                throw new ArgumentException();
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
            int numFeatures = options.UseFeatureHashing ? (2 << options.HashBits) : featureMap.Count;

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
                Stream reader = new FileStream(cacheFile, FileMode.Open, FileAccess.Read);
                using (GZipStream compressedReader = new GZipStream(reader, CompressionMode.Decompress))
                { 
                    while (reader.Position < reader.Length && (cachedInstance = binaryFormatter.Deserialize(compressedReader) as Instance) != null)
                    {
                        trainer.LearnFromOneInstance(cachedInstance.WordsWithFeatures, cachedInstance.LabelledTags);
                        c++;
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
            int numFeatures = options.UseFeatureHashing ? (2 << options.HashBits) : featureMap.Count;

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
                Console.WriteLine(string.Join("\t", new object[] {
                        rTagMap[i],
                        perTagModelCount[i],
                        perTagCorrect[i],
                        perTagCount[i],
                        perTagCorrect[i] * 1.0 / perTagModelCount[i],
                        perTagCorrect[i] * 1.0 / perTagCount[i]
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
                    options.HashBits = int.Parse(args[++i]);
                }
                else if ("--basepath".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    options.BasePath = args[++i];
                }
                else if ("--numiterations".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    options.NumIterations = int.Parse(args[++i]);
                }
                else if ("--data".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    options.DataFile = args[++i];
                }
                else if ("--output".Equals(args[i], StringComparison.OrdinalIgnoreCase))
                {
                    options.Output = args[++i];
                }
                else
                {
                    throw new ArgumentException(args[i]);
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
                string line = null;
                List<int[]> features = new List<int[]>();
                List<int> labelledTags = new List<int>();

                while (!reader.EndOfStream)
                {
                    //read one example
                    while (!string.IsNullOrEmpty((line = reader.ReadLine())))
                    {
                        string[] tokens = Regex.Split(line, @"\s+");

                        //if using feature hashing, dont maintain a feature name to id map
                        //instead use hasing to compute it on the fly
                        if (options.UseFeatureHashing)
                        {
                            features.Add(tokens
                                .Skip(1)
                                .Select(x =>
                                    MurMurHash3.Hash(Encoding.UTF8.GetBytes(x)) & ((2 << options.HashBits) - 1))
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

                        //ugly hack : only affects reported PR stats not final output
                        //if tag is unseen assume it to be O
                        labelledTags.Add(tagMap.ContainsKey(tokens[0]) ? tagMap[tokens[0]] : tagMap["O"]);
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
        public string BasePath { get; set; }
        public bool UseFeatureHashing { get; set; }
        public int HashBits { get; set; }
        public string DataFile { get; set; }
        public int NumIterations { get; set; }
        public string Output { get; set; }
    }
}
