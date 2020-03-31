using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SimpleRegression
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
                .Append(context.Regression.Trainers.Ols());

            var model = pipeline.Fit(data);

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