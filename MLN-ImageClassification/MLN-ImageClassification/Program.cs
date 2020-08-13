using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ImageClassification
{
    class Program
    {
        private static readonly string _imagePath = "..\\..\\..\\Data";
        private static readonly string _savePath = "..\\..\\..\\Model\\hotdog.zip";
        private static readonly string _predictedLabelColumnName = "PredictedLabel";
        private static readonly string _keyColumnName = "LabelAsKey";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Create a DataView containing the image paths and labels
            var input = LoadLabeledImagesFromPath(_imagePath);
            var data = context.Data.LoadFromEnumerable(input);
            data = context.Data.ShuffleRows(data);

            // Load the images and convert the labels to keys to serve as categorical values
            var images = context.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Input.Label), outputColumnName: _keyColumnName)
                .Append(context.Transforms.LoadRawImageBytes(inputColumnName: nameof(Input.ImagePath), outputColumnName: nameof(Input.Image), imageFolder: _imagePath))
                .Fit(data).Transform(data);

            // Split the dataset for training and testing
            var trainTestData = context.Data.TrainTestSplit(images, testFraction: 0.2, seed: 1);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Create an image-classification pipeline and train the model
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = nameof(Input.Image),
                LabelColumnName = _keyColumnName,
                ValidationSet = testData,
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101, // Pretrained DNN
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                TestOnTrainSet = false
            };

            var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue(_predictedLabelColumnName));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluate the model and show the results
            var predictions = model.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumnName: _keyColumnName, predictedLabelColumnName: _predictedLabelColumnName);

            Console.WriteLine();
            Console.WriteLine($"Macro accuracy = {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"Micro accuracy = {metrics.MicroAccuracy:P2}");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine();

            // Save the model
            Console.WriteLine();
            Console.WriteLine("Saving the model...");
            context.Model.Save(model, trainData.Schema, _savePath);
        }

        private static List<Input> LoadLabeledImagesFromPath(string path)
        {
            var images = new List<Input>();
            var directories = Directory.EnumerateDirectories(path);

            foreach(var directory in directories)
            {
                var files = Directory.EnumerateFiles(directory);

                images.AddRange(files.Select(x => new Input {
                    ImagePath = Path.GetFullPath(x),
                    Label = Path.GetFileName(directory)
                }));
            }

            return images;
        }
    }

    public class Input
    {
        public byte[] Image;
        public string ImagePath;
        public string Label;
    }

    public class Output
    {
        public float[] Score;
        public string PredictedLabel;
    }
}