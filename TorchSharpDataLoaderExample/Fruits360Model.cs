namespace TorchSharpDataLoaderExample
{

    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    class Fruits360Model : Module
    {
        private Module layer1 = Sequential(
            Conv2d(3, 32, 3),
            ReLU(),
            MaxPool2d(2, 2));

        private Module layer2 = Sequential(
            Conv2d(32, 64, 3),
            ReLU(),
            MaxPool2d(2, 2));

        private Module layer3 = Sequential(
            Conv2d(64, 64, 3),
            ReLU(),
            MaxPool2d(2, 2));

        private Module fc = Sequential(
            Flatten(),
            Linear(6400, 1024),
            ReLU(),
            Dropout(),
            Linear(1024, 625),
            ReLU(),
            Linear(625, 131));
        public Fruits360Model(Device? device) : base("fruits360")
        {
            RegisterComponents();
            to(device ?? CPU);
        }

        public override Tensor forward(Tensor t)
        {
            t = layer1.forward(t);
            t = layer2.forward(t);
            t = layer3.forward(t);
            t = fc.forward(t);
            return LogSoftmax(0).forward(t);
        }
    }
}