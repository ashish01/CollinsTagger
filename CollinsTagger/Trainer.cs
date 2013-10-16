using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace CollinsTagger
{
    public class Trainer
    {
        private int numFeatures;
        private int numTags;

        private float[,] w_alphaTagFeature;
        private float[,] w_alphaTagPreviousTag;
        private float[,] c_alphaTagFeature;
        private float[,] c_alphaTagPreviousTag;

        private int[] decodedTagsBuffer;
        private float[,] latticeBuffer;
        private int[,] backPointerBuffer;

        private int c;
        private int showFactor;
        private double loss;
        private int instances;

        public Trainer(int numTags, int numFeatures)
        {
            this.numFeatures = numFeatures;
            this.numTags = numTags;

            w_alphaTagFeature = new float[numTags, numFeatures];
            w_alphaTagPreviousTag = new float[numTags, numTags];

            c_alphaTagFeature = new float[numTags, numFeatures];
            c_alphaTagPreviousTag = new float[numTags, numTags];

            decodedTagsBuffer = new int[Viterbi.MAX_WORDS];
            latticeBuffer = new float[Viterbi.MAX_WORDS, numTags];
            backPointerBuffer = new int[Viterbi.MAX_WORDS, numTags];

            c = 1;
            showFactor = 1;
            loss = 0;
            instances = 0;
        }

        public int[] LearnFromOneInstance(int[][] wordsWithActiveFeatures, int[] labelledTags)
        {
            // averaged percetron code : http://ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf

            int numWords = wordsWithActiveFeatures.Length;

            Viterbi.Decode(
                wordsWithActiveFeatures,
                w_alphaTagFeature,
                w_alphaTagPreviousTag,
                latticeBuffer,
                backPointerBuffer,
                numTags,
                decodedTagsBuffer);

            //find mistakes
            int diff = 0;
            for (int t = 0; t < numWords; t++)
            {
                if (decodedTagsBuffer[t] != labelledTags[t])
                    diff++;
            }

            loss += diff * 1.0 / numWords;
            instances++;

            if (c % showFactor == 0)
            {
                showFactor *= 2;
                Console.WriteLine("Average loss {0} = {1}", c, loss / instances);
                instances = 0;
                loss = 0;
            }

            //do perceptron update
            if (diff > 0)
            {
                for (int t = 0; t < numWords; t++)
                {
                    int dt = decodedTagsBuffer[t]; //decoded tag
                    int lt = labelledTags[t]; //labelled tag
                    if (decodedTagsBuffer[t] != labelledTags[t])
                    {
                        //update W
                        foreach (int f in wordsWithActiveFeatures[t])
                        {
                            w_alphaTagFeature[dt, f] -= 1;
                            c_alphaTagFeature[dt, f] -= c;

                            w_alphaTagFeature[lt, f] += 1;
                            c_alphaTagFeature[lt, f] += c;
                        }
                    }

                    if (t > 0)
                    {
                        int dpt = decodedTagsBuffer[t - 1]; //decoded previous tag
                        int lpt = labelledTags[t - 1]; //labelled previous tag

                        if (dt != lt || dpt != lpt)
                        {
                            w_alphaTagPreviousTag[dt, dpt] -= 1;
                            c_alphaTagPreviousTag[dt, dpt] -= c;

                            w_alphaTagPreviousTag[lt, lpt] += 1;
                            c_alphaTagPreviousTag[lt, lpt] += c;
                        }
                    }
                }
                c += 1;
            }

            return decodedTagsBuffer;
        }

        public Model GetModel()
        {
            float[,] tf = new float[numTags, numFeatures];
            float[,] tpt = new float[numTags, numTags];

            for (int t = 0; t < numTags; t++)
            {
                for (int f = 0; f < numFeatures; f++)
                {
                    tf[t, f] = w_alphaTagFeature[t, f] - c_alphaTagFeature[t, f] / c;
                }

                for (int tt = 0; tt < numTags; tt++)
                {
                    tpt[t, tt] = w_alphaTagPreviousTag[t, tt] - c_alphaTagPreviousTag[t, tt] / c;
                }
            }

            return new Model()
            {
                alphaTagFeature = tf,
                alphaTagPreviousTag = tpt
            };
        }
    }

    public class Model
    {
        public float[,] alphaTagFeature { get; set; }
        public float[,] alphaTagPreviousTag { get; set; }
    }
}
