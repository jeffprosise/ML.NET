using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;

namespace AutoML
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\pacific-heights.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the data
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Create an experiment
            var settings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 1800, // 30 minutes max
                OptimizingMetric = RegressionMetric.RSquared,
                CacheDirectory = null
            };

            var experiment = context.Auto().CreateRegressionExperiment(settings);

            // Run the experiment
            Console.WriteLine("Running the experiment...");
            var result = experiment.Execute(data);

            RegressionMetrics metrics = result.BestRun.ValidationMetrics;
            Console.WriteLine($"R2 score: {metrics.RSquared:0.##}");
            Console.WriteLine();

            // Use the best model to make a prediction
            var predictor = context.Model.CreatePredictionEngine<Input, Output>(result.BestRun.Model);

            var input = new Input
            {
                Bathrooms = 1.0f,
                Bedrooms = 1.0f,
                TotalRooms = 3.0f,
                FinishedSquareFeet = 653.0f,
                UseCode = "Condominium",
                LastSoldPrice = 0.0f
            };

            var prediction = predictor.Predict(input);

            Console.WriteLine($"Predicted price: ${prediction.Price:n0}; Actual price: $665,000");
            Console.WriteLine();
        }
    }

    public class Input
    {
        [LoadColumn(1)]
        public float Bathrooms;

        [LoadColumn(2)]
        public float Bedrooms;

        [LoadColumn(3)]
        public float FinishedSquareFeet;

        [LoadColumn(5), ColumnName("Label")]
        public float LastSoldPrice;

        [LoadColumn(9)]
        public float TotalRooms;

        [LoadColumn(10)]
        public string UseCode;
    }

    public class Output
    {
        [ColumnName("Score")]
        public float Price;
    }
}
