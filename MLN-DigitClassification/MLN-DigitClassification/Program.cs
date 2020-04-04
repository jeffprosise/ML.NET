using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace DigitClassification
{
    class Program
    {
        static readonly string _trainDataPath = "..\\..\\..\\Data\\mnist-digits-train.csv";
        static readonly string _testDataPath = "..\\..\\..\\Data\\mnist-digits-test.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            var trainData = context.Data.LoadFromTextFile<Input>(_trainDataPath, hasHeader: false, separatorChar: ',');
            var testData = context.Data.LoadFromTextFile<Input>(_testDataPath, hasHeader: false, separatorChar: ',');

            // Build and train the model
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label", "Digit", keyOrdinality: KeyOrdinality.ByValue)
                .Append(context.Transforms.Concatenate("Features", "PixelValues"))
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(context.Transforms.Conversion.MapKeyToValue("Digit", "Label"));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine();
            Console.WriteLine($"Macro accuracy = {(metrics.MacroAccuracy * 100):0.##}%");
            Console.WriteLine($"Micro accuracy = {(metrics.MicroAccuracy * 100):0.##}%");
            Console.WriteLine();

            // Use the model to make a prediction
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            var input = new Input
            {
                PixelValues = new float[]
                {
                    0, 0,  0,  0, 12,  2, 0, 0,
                    0, 0,  0,  6, 14,  1, 0, 0,
                    0, 0,  4, 16,  7,  8, 0, 0,
                    0, 0, 13,  9,  0, 16, 6, 0,
                    0, 6, 16, 10, 11, 16, 0, 0,
                    0, 0,  5, 10, 13, 16, 0, 0,
                    0, 0,  0,  0,  6, 16, 0, 0,
                    0, 0,  0,  0, 12,  8, 0, 0
                }
            }; // 4

            var prediction = predictor.Predict(input);

            int i = 0;
            foreach (var score in prediction.Score)
            {
                Console.WriteLine($"{i++} - {score:N8}");
            }

            Console.WriteLine();
            int index = prediction.Score.ToList().IndexOf(prediction.Score.Max());
            Console.WriteLine($"Looks like a {index}!");
            Console.WriteLine();
        }
    }

    class Input
    {
        [LoadColumn(0, 63), VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64)]
        public float Digit;
    }

    class Output
    {
        public float[] Score;
    }
}