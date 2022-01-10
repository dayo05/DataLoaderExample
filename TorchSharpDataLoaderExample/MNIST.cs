﻿#define MNIST

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using TorchSharpDataLoaderExample;

using TorchSharp;
using static System.Linq.Enumerable;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


#if MNIST
using var trainDataset = new MNISTReader("dataset", "train");
using var testDataset = new MNISTReader("dataset", "t10k");

using var train = new DataLoader(trainDataset, 128, true, CPU);
using var test = new DataLoader(testDataset, 256, false, CPU);

var model = new Model();

var criterion = functional.cross_entropy_loss();
var optimizer = optim.Adam(model.parameters(), learningRate: 0.01);

Console.WriteLine("Initialized");
foreach (var x in train) { }
foreach(var epoch in Range(1, 20)){
    var avg_cost = 0.0;
    var idx = 0;
    foreach (var d in train)
    {
        Console.Write($"\r{idx++} / {train.Count}");
        optimizer.zero_grad();
        var hypothesis = model.forward(d["data"]);
        var cost = criterion(hypothesis, d["label"]);
        cost.backward();
        optimizer.step();

        avg_cost += cost.ToSingle() / train.Count;
    }
    Console.WriteLine("\r" + avg_cost);
}
class Model : Module
{
    private Module conv1 = Conv2d(1, 32, 3);
    private Module conv2 = Conv2d(32, 64, 3);
    private Module fc1 = Linear(9216, 128);
    private Module fc2 = Linear(128, 10);

    // These don't have any parameters, so the only reason to instantiate
    // them is performance, since they will be used over and over.
    private Module pool1 = MaxPool2d(kernelSize: new long[] { 2, 2 });

    private Module relu1 = ReLU();
    private Module relu2 = ReLU();
    private Module relu3 = ReLU();

    private Module dropout1 = Dropout(0.25);
    private Module dropout2 = Dropout(0.5);

    private Module flatten = Flatten();
    private Module logsm = LogSoftmax(1);

    public Model() : base("h")
    {
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var l11 = conv1.forward(input);
        var l12 = relu1.forward(l11);

        var l21 = conv2.forward(l12);
        var l22 = relu2.forward(l21);
        var l23 = pool1.forward(l22);
        var l24 = dropout1.forward(l23);

        var x = flatten.forward(l24);

        var l31 = fc1.forward(x);
        var l32 = relu3.forward(l31);
        var l33 = dropout2.forward(l32);

        var l41 = fc2.forward(l33);

        return logsm.forward(l41);
    }
}
#endif