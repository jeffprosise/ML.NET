using Microsoft.ML;
using Microsoft.ML.Data;
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
        private static readonly string _hotDogImagePath = "..\\..\\..\\Data\\predict\\1007.jpg";
        private static readonly string _sushiImagePath = "..\\..\\..\\Data\\predict\\1008.jpg";
        private static readonly string _modelPath = "..\\..\\..\\Model\\tensorflow_inception_graph.pb";
        private static readonly string _savePath = "..\\..\\..\\Model\\hotdog.zip";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the training data
            var trainingData = new List<ImageData>();
            LoadImageData(trainingData, _hotDogTrainImagesPath, "hotdog");
            LoadImageData(trainingData, _notHotDogTrainImagesPath, "nothotdog");

            var pipeline = context.Transforms.Conversion.MapValueToKey(outputColumnName: "labelTokey", inputColumnName: "Label")
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: _hotDogTrainImagesPath, inputColumnName: "ImagePath"))
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: _notHotDogTrainImagesPath, inputColumnName: "ImagePath"))
                .Append(context.Transforms.ResizeImages(outputColumnName: "input", inputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel(_modelPath)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true)
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "labelTokey", featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel")));

            // Train the model
            Console.WriteLine("Training the model...");
            var data = context.Data.LoadFromEnumerable<ImageData>(trainingData); // Create IDataView from IEnumerable
            var model = pipeline.Fit(data);
            Console.WriteLine();

            //var predictions = model.Transform(data);
            //var imageData = context.Data.CreateEnumerable<ImageData>(data, false, true);
            //var imagePredictionData = context.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);

            // Make a pair of predictions
            var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            var hotdog = new ImageData { ImagePath = _hotDogImagePath };
            var result = predictor.Predict(hotdog);
            Console.WriteLine(result.PredictedLabelValue == "hotdog" ? "It's a hot dog!" : "Not a hot dog");

            var sushi = new ImageData { ImagePath = _sushiImagePath };
            result = predictor.Predict(sushi);
            Console.WriteLine(result.PredictedLabelValue == "hotdog" ? "It's a hot dog!" : "Not a hot dog");

            // Save the model
            Console.WriteLine();
            Console.WriteLine("Saving the model");
            context.Model.Save(model, data.Schema, _savePath);
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
