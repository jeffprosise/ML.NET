using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace MLN_SpamDetection
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\ham-spam.csv";

        static readonly string[] _samples =
        {
            "Excellent! We will proceed with outlines, if for no other reason than it helps us think hard about how these modules should be structured.",
            "Can you attend a meeting in Atlanta on the 16th? It's important to get all the relevant players together to discuss next year's marketing plan.",
            "Why pay more for expensive meds when you can order them online and save $$$?"
        };

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the data
            var data = context.Data.LoadFromTextFile<EmailInput>(_path, hasHeader: true, separatorChar: ',');

            // Split the data into a training set and a test set
            var trainTestData = context.BinaryClassification.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Build and train the model
            var pipeline = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "Text")
                .Append(context.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.Auc:P2}");
            Console.WriteLine($"F1: {metrics.F1Score:P2}");

            // Use the model to make predictions
            var predictor = model.CreatePredictionEngine<EmailInput, EmailPrediction>(context);

            foreach (var sample in _samples)
            {
                var input = new EmailInput { Text = sample };
                var prediction = predictor.Predict(input);

                Console.WriteLine();
                Console.WriteLine($"{input.Text}");
                Console.WriteLine($"Spam score: {prediction.Probability}");
                Console.WriteLine($"Classification: {(Convert.ToBoolean(prediction.Prediction) ? "Spam" : "Not spam")}");
            }

            Console.WriteLine();
        }
    }

    public class EmailInput
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool IsSpam;

        [LoadColumn(1)]
        public string Text;
    }

    public class EmailPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
