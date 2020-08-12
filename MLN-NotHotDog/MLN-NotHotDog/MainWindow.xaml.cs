using Microsoft.ML;
using Microsoft.Win32;
using System;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;

namespace MLN_NotHotDog
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private PredictionEngine<ImageData, ImagePrediction> _predictor;
        private static readonly string _modelPath = "..\\..\\..\\Model\\hotdog.zip";

        public MainWindow()
        {
            InitializeComponent();
            this.Loaded += OnLoaded;
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            // Load a trained ML.NET model
            var context = new MLContext(seed: 0);
            var model = context.Model.Load(_modelPath, out DataViewSchema schema);
            _predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        }

        private void OnSelectImageButtonClicked(object sender, RoutedEventArgs e)
        {
            // Let the user select an image file
            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "JPEG Files (*.jpg;*.jpeg)|*.jpg;*.jpeg|PNG Files (*.png)|*.png|All Files (*.*)|*.*";
            dialog.FilterIndex = 3;

            if (dialog.ShowDialog() == true)
            {
                string path = dialog.FileName;

                // Show the image
                LoadedImage.Source = new BitmapImage(new Uri(path));

                try
                {
                    // Use ML.NET to determine whether the image contains a hot dog
                    var image = new ImageData { ImagePath = path };
                    var result = _predictor.Predict(image);
                    var label = result.PredictedLabel;
                    var probability = result.Score.Max();

                    if (String.Compare(label, "hotdog", true) == 0)
                    {
                        MessageBox.Show($"It's a hot dog! ({probability:P1})");
                    }
                    else
                    {
                        MessageBox.Show($"Not a hot dog. Looks more like {label}.");
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
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
}