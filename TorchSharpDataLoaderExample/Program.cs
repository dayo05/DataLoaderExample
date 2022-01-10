using TorchSharpDataLoaderExample;

using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static System.Linq.Enumerable;

#if FRUITS
using var trainDataset = new Fruits360("/home/dayo/datasets/fruits-360/", true, CUDA);
using var testDataset = new Fruits360("/home/dayo/datasets/fruits-360/", false, CUDA);

using var train = new DataLoader(trainDataset, 256, true, CUDA);
using var test = new DataLoader(testDataset, 512, false, CUDA);

var model = new Fruits360Model(CUDA);
var optimizer = optim.Adam(model.parameters(), learningRate: 0.01);

foreach (var epoch in Range(1, 1000))
{
    model.Train();
    var batchId = 1;
    Console.WriteLine($"Epoch{epoch} running");
    foreach (var x in train)
    {
        optimizer.zero_grad();

        var prediction = model.forward(x["image"]);
        var output = functional.nll_loss(reduction: Reduction.Mean)(prediction, x["label"]);
        
        output.backward();
        optimizer.step();
        
        Console.Write($"\rTrain: epoch {epoch} {batchId * 1.0 / train.Count:P2} [{batchId} / {train.Count}] Loss: {output.ToSingle():F9}");
        batchId++;
        
        prediction.Dispose();
        output.Dispose();
        GC.Collect();
    }

    using (no_grad())
    {
        model.Eval();
        
        var testLoss = 0.0;
        var correct = 0;
        var idx = 0;
        foreach (var x in test)
        {
            idx++;
            Console.Write($"\rTest running: {idx * 1.0 / test.Count:P2}");
            var prediction = model.forward(x["image"]);
            var output = functional.nll_loss(reduction: Reduction.Sum)(prediction, x["label"]);
            testLoss += output.ToSingle();

            var pred = prediction.argmax(1);
            correct += pred.eq(x["label"]).sum().ToInt32();
            pred.Dispose();
            prediction.Dispose();
            output.Dispose();
            GC.Collect();
        }
        Console.WriteLine(
            $"\rTest set: Average loss {(testLoss / testDataset.Count):F9} | Accuracy {((double) correct / testDataset.Count):P2}");
    }
}
#endif