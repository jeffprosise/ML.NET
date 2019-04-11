using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLN_ImageClassification
{
    class Program
    {
        private static readonly string _hotDogTrainImagesPath = "..\\..\\..\\Data\\train\\hot-dog";
        private static readonly string _notHotDogTrainImagesPath = "..\\..\\..\\Data\\train\\not-hot-dog";
        private static readonly string _hotDogTestImagesPath = "..\\..\\..\\Data\\test\\hot-dog";
        private static readonly string _notHotDogTestImagesPath = "..\\..\\..\\Data\\test\\not-hot-dog";
        private static readonly string _hotDogImagePath = "..\\..\\..\\Data\\predict\\1007.jpg";
        private static readonly string _sushiImagePath = "..\\..\\..\\Data\\predict\\1008.jpg";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);






        }
    }

    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }

    public class ImagePrediction
    {
        public float[] Score;
        public string PredictedLabelValue;
    }
}
