using System;
using System.Collections.Generic;
using System.IO;
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

            // Load the training data
            var trainingData = new List<ImageData>();
            var files = Directory.EnumerateFiles(_hotDogTrainImagesPath);

            foreach(var file in files)
            {
                var imageData = new ImageData
                {
                    ImagePath = Path.GetFullPath($"{_hotDogTrainImagesPath}\\{file}"),
                    Label = "hotdog"
                };

                trainingData.Add(imageData);
            }

            files = Directory.EnumerateFiles(_notHotDogTrainImagesPath);

            foreach (var file in files)
            {
                var imageData = new ImageData
                {
                    ImagePath = Path.GetFullPath($"{_notHotDogTrainImagesPath}\\{file}"),
                    Label = "nothotdog"
                };

                trainingData.Add(imageData);
            }



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
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }

    //public class ImagePrediction
    //{
    //    public float[] Score;
    //    public string PredictedLabelValue;
    //}

    public struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}
