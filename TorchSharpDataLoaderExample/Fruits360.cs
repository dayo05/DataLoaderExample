namespace TorchSharpDataLoaderExample
{

    using SixLabors.ImageSharp;
    using SixLabors.ImageSharp.Advanced;
    using SixLabors.ImageSharp.Formats.Jpeg;
    using SixLabors.ImageSharp.PixelFormats;

    using static TorchSharp.torch;

    public class Fruits360 : Dataset
    {
        private List<string> Labels = new();
        private List<string> images = new();
        private Device device;
        public Fruits360(string root, bool isTrain, Device device = null)
        {
            this.device = device ?? CPU;
            root += isTrain ? "Training" : "Test";
            foreach (var x in Directory.GetDirectories(root))
                images.AddRange(Directory.GetFiles(x));
            Labels.AddRange(Directory.GetDirectories(root).Select(x => x.Split('/')[^1]));
        }

        public override long Count => images.Count;

        public override Dictionary<string, Tensor> GetTensor(long index)
        {
            var image = Image.Load<Rgb24>(images[(int)index], new JpegDecoder());
            using var r = tensor(image.GetPixelMemoryGroup()[0].Span.ToArray().Select(x => x.R / 255.0f).ToList(),
                new long[] { 1, 100, 100 }).to(device);
            using var g = tensor(image.GetPixelMemoryGroup()[0].Span.ToArray().Select(x => x.G / 255.0f).ToList(),
                new long[] { 1, 100, 100 }).to(device);
            using var b = tensor(image.GetPixelMemoryGroup()[0].Span.ToArray().Select(x => x.B / 255.0f).ToList(),
                new long[] { 1, 100, 100 }).to(device);
            return new()
            {
                { "image", cat(new List<Tensor> { r, g, b }, 0) },
                { "label", tensor(Labels.IndexOf(images[(int)index].Split('/')[^2]), ScalarType.Int64) }
            };
        }
    }
}