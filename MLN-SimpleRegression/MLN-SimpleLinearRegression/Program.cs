using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace MLN_SimpleLinearRegression
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\poverty.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Train the model
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "BirthRate")
                .Append(context.Transforms.Concatenate("Features", "PovertyRate"))
                .Append(context.Regression.Trainers.OnlineGradientDescent(learningRate: 0.01f, numIterations: 1000));

            var model = pipeline.Fit(data);

            // Compute the model's R2 score
            var predictions = model.Transform(data);
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine($"R2 score: {metrics.RSquared:0.##}");

            // Use the model to make a prediction
            var predictor = model.CreatePredictionEngine<Input, Output>(context);
            var input = new Input { PovertyRate = 19.7f };
            var prediction = predictor.Predict(input);

            Console.WriteLine($"Predicted birth rate: {prediction.BirthRate:0.##}");
            Console.WriteLine($"Actual birth rate: 58.10");
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public float PovertyRate;

        [LoadColumn(5)]
        public float BirthRate;
    }

    public class Output
    {
        [ColumnName("Score")]
        public float BirthRate;
    }
}
