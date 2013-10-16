using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace CollinsTagger
{
    public class Tagger
    {
        private float[,] alphaTagFeature;
        private float[,] alphaTagPreviousTag;

        private float[,] latticeBuffer;
        private int[,] backPointerBuffer;

        private readonly int numTags;

        public Tagger(int numTags, float[,] alphaTagFeature, float[,] alphaTagPreviousTag)
        {
            this.alphaTagFeature = alphaTagFeature;
            this.alphaTagPreviousTag = alphaTagPreviousTag;
            this.numTags = numTags;

            //we will decode a max of 200 word sequences
            this.latticeBuffer = new float[Viterbi.MAX_WORDS, numTags];
            this.backPointerBuffer = new int[Viterbi.MAX_WORDS, numTags];
        }

        public void Label(int[][] wordsWithActiveFeatures, int[] tags)
        {
            Viterbi.Decode(wordsWithActiveFeatures, alphaTagFeature, alphaTagPreviousTag, latticeBuffer, backPointerBuffer, numTags, tags);
        }
    }
}
