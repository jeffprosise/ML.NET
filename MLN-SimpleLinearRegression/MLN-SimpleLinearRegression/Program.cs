using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SimpleLinearRegression
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\poverty.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Train the model
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            var pipeline = context.Transforms.Concatenate("Features", "PovertyRate")
                .Append(context.Regression.Trainers.OnlineGradientDescent(learningRate: 0.01f, numberOfIterations: 1000));

            var model = pipeline.Fit(data);

            // Compute the model's R2 score
            var predictions = model.Transform(data);
            var metrics = context.Regression.Evaluate(predictions);
            Console.WriteLine($"R2 score: {metrics.RSquared:0.##}");

            // Use the model to make a prediction
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
            var input = new Input { PovertyRate = 19.7f };
            var prediction = predictor.Predict(input);

            Console.WriteLine($"Predicted birth rate: {prediction.BirthRate:0.##}");
            Console.WriteLine($"Actual birth rate: 58.10");
            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public float PovertyRate;

        [LoadColumn(5), ColumnName("Label")]
        public float BirthRate;
    }

    public class Output
    {
        [ColumnName("Score")]
        public float BirthRate;
    }
}