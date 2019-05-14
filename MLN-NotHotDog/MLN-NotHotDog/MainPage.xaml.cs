using Microsoft.ML;
using System;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Popups;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace MLN_NotHotDog
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private PredictionEngine<ImageData, ImagePrediction> _predictor;
        private static readonly string _modelPath = "ms-appx:///Model/hotdog.zip";

        public MainPage()
        {
            this.InitializeComponent();
        }

        protected async override void OnNavigatedTo(NavigationEventArgs e)
        {
            base.OnNavigatedTo(e);

            // Load a trained ML.NET model
            var context = new MLContext(seed: 0);
            var model = context.Model.Load(await GetFilePathAsync(_modelPath), out DataViewSchema schema);
            _predictor = context.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
        }

        private async void OnSelectImageButtonClicked(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".jpeg");
            picker.FileTypeFilter.Add(".png");

            var file = await picker.PickSingleFileAsync();

            if (file != null)
            {
                string path;
                byte[] buffer;

                using (IRandomAccessStream stream = await file.OpenAsync(FileAccessMode.Read))
                {
                    // Show the image
                    var image = new BitmapImage();
                    await image.SetSourceAsync(stream);
                    LoadedImage.Source = image;

                    // Read the image into a byte array
                    stream.Seek(0L);
                    var bytes = new byte[stream.Size];
                    await stream.ReadAsync(bytes.AsBuffer(), (uint)stream.Size, InputStreamOptions.None);
                    buffer = bytes;

                    // Write the image to a storage file
                    var folder = ApplicationData.Current.TemporaryFolder;
                    var tempfile = await folder.CreateFileAsync(file.Name, CreationCollisionOption.ReplaceExisting);
                    await FileIO.WriteBytesAsync(tempfile, buffer);

                    // Save the path to the storage file
                    path = tempfile.Path;
                }

                try
                {
                    // Use ML.NET to determine whether the image contains a hot dog
                    var image = new ImageData { ImagePath = path };
                    var result = _predictor.Predict(image);
                    var label = result.PredictedLabelValue;
                    var probability = result.Score.Max();

                    if (String.Compare(label, "hotdog", true) == 0 && probability > 0.90)
                    {
                        await new MessageDialog("It's a hot dog!").ShowAsync();
                    }
                    else
                    {
                        await new MessageDialog("Not a hot dog").ShowAsync();
                    }
                }
                catch (Exception ex)
                {
                    await new MessageDialog(ex.Message).ShowAsync();
                }
            }
        }

        private static async Task<string> GetFilePathAsync(string path)
        {
            var originalFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri(path));
            var storageFolder = ApplicationData.Current.TemporaryFolder;
            var localFile = await originalFile.CopyAsync(storageFolder, originalFile.Name, NameCollisionOption.ReplaceExisting);
            return localFile.Path;
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
        public string PredictedLabelValue;
    }
}