using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using System;
using System.Collections.Generic;
using System.IO;

namespace MLN_ImageClassification
{
    class Program
    {
        private static readonly string _hotDogTrainImagesPath = "..\\..\\..\\Data\\train\\hot-dog";
        private static readonly string _notHotDogTrainImagesPath = "..\\..\\..\\Data\\train\\not-hot-dog";
        //private static readonly string _hotDogTestImagesPath = "..\\..\\..\\Data\\test\\hot-dog";
        //private static readonly string _notHotDogTestImagesPath = "..\\..\\..\\Data\\test\\not-hot-dog";
        //private static readonly string _hotDogImagePath = "..\\..\\..\\Data\\predict\\1007.jpg";
        //private static readonly string _sushiImagePath = "..\\..\\..\\Data\\predict\\1008.jpg";
        private static readonly string _modelPath = "..\\..\\..\\Model\\tensorflow_inception_graph.pb";
        private static readonly string _labelToKey = "labelTokey";
        private static readonly string _imageReal = "ImageReal";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the training data
            var trainingData = new List<ImageData>();
            LoadImageData(trainingData, _hotDogTrainImagesPath, "hotdog");
            LoadImageData(trainingData, _notHotDogTrainImagesPath, "nothotdog");

            // Build the model
            var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: _labelToKey, inputColumnName: "Label")
                .Append(context.Transforms.LoadImages(_hotDogTrainImagesPath, (_imageReal, nameof(ImageData.ImagePath)))
                .Append(context.Transforms.LoadImages(_notHotDogTrainImagesPath, (_imageReal, nameof(ImageData.ImagePath)))
                .Append(context.Transforms.ResizeImages(outputColumnName: _imageReal, imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: _imageReal))
                .Append(context.Transforms.ExtractPixels(new ImagePixelExtractingEstimator.ColumnOptions(name: "input", inputColumnName: _imageReal, interleave: InceptionSettings.ChannelsLast, offset: InceptionSettings.Mean)))
                .Append(context.Transforms.ScoreTensorFlowModel(modelLocation: _modelPath, outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }))
                //.Append(context.BinaryClassification.Trainers.LogisticRegression(labelColumnName: _labelToKey, featureColumnName: "softmax2_pre_activation"))
                .Append(context.MulticlassClassification.Trainers.LogisticRegression(labelColumnName: _labelToKey, featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))));

            // Train the model
            Console.WriteLine("Training the model...");
            var data = context.Data.LoadFromEnumerable<ImageData>(trainingData);
            var model = pipeline.Fit(data);
            var predictions = model.Transform(data);

            var imageData = context.Data.CreateEnumerable<ImageData>(data, false, true);
            var imagePredictionData = context.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);

            // TODO: Evaluate the model


            // TODO: Save the model


        }

        private static void LoadImageData(List<ImageData> images, string path, string label)
        {
            var files = Directory.EnumerateFiles(path);

            foreach (var file in files)
            {
                var imageData = new ImageData
                {
                    ImagePath = Path.GetFullPath($"{path}\\{file}"),
                    Label = label
                };

                images.Add(imageData);
            }
        }
    }

    public class ImageData
    {
        public string ImagePath;
        public string Label;
    }

    //public class ImagePrediction
    //{
    //    [ColumnName("PredictedLabel")]
    //    public bool Prediction { get; set; }

    //    public float Probability { get; set; }
    //}

    public class ImagePrediction
    {
        public float[] Score;
        public string PredictedLabelValue;
    }

    public struct InceptionSettings
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}
