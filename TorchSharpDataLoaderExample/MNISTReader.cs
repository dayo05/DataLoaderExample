﻿using TorchSharp.torchvision;
using static TorchSharp.torch;

namespace TorchSharpDataLoaderExample
{
    public class BigEndianReader
    {
        public BigEndianReader(BinaryReader baseReader)
        {
            mBaseReader = baseReader;
        }

        public int ReadInt32()
            => BitConverter.ToInt32(ReadBigEndianBytes(4), 0);

        public byte[] ReadBigEndianBytes(int count)
        {
            byte[] bytes = new byte[count];
            for (int i = count - 1; i >= 0; i--)
                bytes[i] = mBaseReader.ReadByte();

            return bytes;
        }

        public byte[] ReadBytes(int count)
            => mBaseReader.ReadBytes(count);

        public void Close()
            => mBaseReader.Close();

        public Stream BaseStream
        {
            get { return mBaseReader.BaseStream; }
        }

        private BinaryReader mBaseReader;
    }

    /// <summary>
    /// Data reader utility for datasets that follow the MNIST data set's layout:
    ///
    /// A number of single-channel (grayscale) images are laid out in a flat file with four 32-bit integers at the head.
    /// The format is documented at the bottom of the page at: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public sealed class MNISTReader : Dataset
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="path">Path to the folder containing the image files.</param>
        /// <param name="prefix">The file name prefix, either 'train' or 't10k' (the latter being the test data set).</param>
        /// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
        /// <param name="transform"></param>
        public MNISTReader(string path, string prefix, Device device = null, ITransform transform = null)
        {
            // The MNIST data set is small enough to fit in memory, so let's load it there.

            this.transform = transform;

            var dataPath = Path.Combine(path, prefix + "-images.idx3-ubyte");
            var labelPath = Path.Combine(path, prefix + "-labels.idx1-ubyte");

            var count = -1;
            var height = 0;
            var width = 0;

            byte[] dataBytes;
            byte[] labelBytes;

            using (var file = File.Open(dataPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new BinaryReader(file))
            {
                var reader = new BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                count = reader.ReadInt32();

                height = reader.ReadInt32();
                width = reader.ReadInt32();

                // Read all the data into memory.
                dataBytes = reader.ReadBytes(height * width * count);
            }

            using (var file = File.Open(labelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var rdr = new BinaryReader(file))
            {
                var reader = new BigEndianReader(rdr);
                var x = reader.ReadInt32(); // Magic number
                var lblcnt = reader.ReadInt32();

                if (lblcnt != count) throw new InvalidDataException("Image data and label counts are different.");

                // Read all the data into memory.
                labelBytes = reader.ReadBytes(lblcnt);
            }

            var imgSize = height * width;

            // Go through the data and create tensors
            for (var i = 0; i < count; i++)
            {
                var imgStart = i * imgSize;

                data.Add(tensor(dataBytes[imgStart..(imgStart+imgSize)].Select(b => b/256.0f).ToArray(), new long[] { width, height }).unsqueeze(0));
                labels.Add(tensor(labelBytes[i], int64));
            }
        }

        private ITransform transform;

        public override long Count => data.Count;

        private List<Tensor> data = new();
        private List<Tensor> labels = new();

        public override void Dispose()
        {
            data.ForEach(d => d.Dispose());
            labels.ForEach(d => d.Dispose());
        }

        public override Dictionary<string, Tensor> GetTensor(long index)
            => new() {
                { "data", transform is not null ? transform.forward(data[(int)index]) : data[(int)index] },
                { "label", labels[(int)index] }
            };
    }
}
