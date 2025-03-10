using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CollinsTagger
{
    public static class Viterbi
    {
        //global maximum length of sequence
        public const int MAX_WORDS = 200;

        //dont make any heap memory allocations
        public static void Decode(
            int[][] wordsWithActiveFeatures,
            float[,] alphaTagFeature,
            float[,] alphaTagPreviousTag,
            float[,] lattice,
            int[,] backPointers,
            int numTags,
            int[] decodedTags)
        {
            int numWords = wordsWithActiveFeatures.Length;

            // Check if we have a valid sequence and tags to process
            if (numWords == 0 || numTags == 0)
            {
                return;
            }

            //find the best tag for first word
            //this score is only dependent on the word features
            //Begining of sentence / end of sentence is encoded as features
            for (int t = 0; t < numTags; t++)
            {
                lattice[0, t] = 0;
                foreach (var feature in wordsWithActiveFeatures[0])
                {
                    lattice[0, t] += alphaTagFeature[t, feature];
                    backPointers[0, t] = -1;
                }
            }

            //find the best tag for rest of the words
            //this score is dependent on features of this word
            //and the previous tag
            for (int w = 1; w < numWords; w++)
            {
                for (int currentTag = 0; currentTag < numTags; currentTag++)
                {
                    //calculate the tag features score only once
                    float tagFeatureScore = 0;
                    foreach (var feature in wordsWithActiveFeatures[w])
                        tagFeatureScore += alphaTagFeature[currentTag, feature];

                    bool first = true;
                    float maxScore = 0;
                    int prevTag = 0;

                    //now go through all the previous tags and calculate candidate score
                    for (int previousTag = 0; previousTag < numTags; previousTag++)
                    {
                        float candidateScore =
                            tagFeatureScore +
                            alphaTagPreviousTag[currentTag, previousTag] +
                            lattice[w - 1, previousTag];

                        if (first)
                        {
                            first = false;
                            maxScore = candidateScore;
                            prevTag = previousTag;
                        }

                        if (candidateScore > maxScore)
                        {
                            maxScore = candidateScore;
                            prevTag = previousTag;
                        }
                    }

                    lattice[w, currentTag] = maxScore;
                    backPointers[w, currentTag] = prevTag;
                }
            }

            // Only backtrack if we have words to process
            if (numWords > 0 && numTags > 0)
            {
                //backtrack back pointers
                float lastScore = lattice[numWords - 1, 0];
                int lastTag = 0;
                for (int t = 1; t < numTags; t++)
                {
                    if (lattice[numWords - 1, t] > lastScore)
                    {
                        lastScore = lattice[numWords - 1, t];
                        lastTag = t;
                    }
                }

                decodedTags[numWords - 1] = lastTag;
                for (int w = numWords - 1; w > 0; w--)
                {
                    decodedTags[w - 1] = backPointers[w, decodedTags[w]];
                }
            }
        }
    }
}
