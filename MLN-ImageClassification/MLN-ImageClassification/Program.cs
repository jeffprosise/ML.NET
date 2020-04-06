using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification
{
    class Program
    {
        private static readonly string _hotDogTrainImagesPath = "..\\..\\..\\Data\\train\\hotdog";
        private static readonly string _pizzaTrainImagesPath = "..\\..\\..\\Data\\train\\pizza";
        private static readonly string _sushiTrainImagesPath = "..\\..\\..\\Data\\train\\sushi";
        private static readonly string _testImagesPath = "..\\..\\..\\Data\\test";
        private static readonly string _modelPath = "..\\..\\..\\Model\\tensorflow_inception_graph.pb";
        private static readonly string _savePath = "..\\..\\..\\Model\\hotdog.zip";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the training data
            var trainingData = new List<ImageData>();
            LoadImageData(trainingData, Path.GetFullPath(_hotDogTrainImagesPath), "hotdog");
            LoadImageData(trainingData, Path.GetFullPath(_pizzaTrainImagesPath), "pizza");
            LoadImageData(trainingData, Path.GetFullPath(_sushiTrainImagesPath), "sushi");

            var pipeline = context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_hotDogTrainImagesPath), inputColumnName: "ImagePath")
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_pizzaTrainImagesPath), inputColumnName: "ImagePath"))
                .Append(context.Transforms.LoadImages(outputColumnName: "input", imageFolder: Path.GetFullPath(_sushiTrainImagesPath), inputColumnName: "ImagePath"))
                .Append(context.Transforms.ResizeImages(outputColumnName: "input", inputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(context.Model.LoadTensorFlowModel(_modelPath)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true)
                .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "Key", inputColumnName: "Label"))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Key", featureColumnName: "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

            // Train the model
            Console.WriteLine("Training the model...");
            var data = context.Data.LoadFromEnumerable<ImageData>(trainingData); // Create IDataView from IEnumerable
            var model = pipeline.Fit(data);
            Console.WriteLine();

            // Make predictions using test images
            var files = Directory.EnumerateFiles(Path.GetFullPath(_testImagesPath));
            var predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

            foreach (var file in files)
            {
                var image = new ImageData { ImagePath = file };
                var result = predictor.Predict(image);
                var label = result.PredictedLabel;
                var probability = result.Score.Max();
                Console.WriteLine($"{Path.GetFileName(file)} - {label} ({probability:P2})");
            }

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
                    ImagePath = file,
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
        public string PredictedLabel;
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