using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

namespace MLN_MultipleRegression
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\pacific-heights.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

            // Load the data
            var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, separatorChar: ',');

            // Split the data into a training set and a test set
            var trainTestData = context.Regression.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // One-hot encode the values in the "UseCode" column and train the model
            var pipeline = context.Transforms.Categorical.OneHotEncoding(inputColumnName: "UseCode", outputColumnName: "UseCodeEncoded")
                .Append(context.Transforms.Concatenate("Features", "UseCodeEncoded", "Bathrooms", "Bedrooms", "TotalRooms", "FinishedSquareFeet"))
                .Append(context.Regression.Trainers.FastForest());

            var model = pipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine($"R2 score: {metrics.RSquared:0.##}");

            // Evaluate the model again using cross-validation
            var scores = context.Regression.CrossValidate(data, pipeline, numFolds: 5);
            var mean = scores.Average(x => x.Metrics.RSquared);
            Console.WriteLine($"Mean cross-validated R2 score: {mean:0.##}");

            // Use the model to make a prediction
            var predictor = model.CreatePredictionEngine<Input, Output>(context);

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