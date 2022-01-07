namespace TorchSharpDataLoaderExample;

using static TorchSharp.torch;
/// <summary>
/// Interface for Dataloader
/// </summary>
public abstract class Dataset : IDisposable
{
    public virtual void Dispose()
    {
    }

    /// <summary>
    /// Size of dataset
    /// </summary>
    public abstract long Count { get; }

    /// <summary>
    /// Get tensor via index
    /// </summary>
    /// <param name="index">Index for tensor</param>
    /// <returns>Tensor for index. You should return dictionary for catenate random amount of tensors.</returns>
    public abstract Dictionary<string, Tensor> GetTensor(long index);
}