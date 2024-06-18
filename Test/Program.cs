Perceptron.Classification classification = new Perceptron.Classification(
    3,
    new int[] { 6, 16, 8 },
    new List<Func<double, double>> { Perceptron.ActFunc.Relu, Perceptron.ActFunc.Relu, Perceptron.ActFunc.Sigmoid }
    );

List<double[]> inputs = new List<double[]>()
{
    new[] { 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 1.0 },
    new[] { 0.0, 1.0, 0.0 },
    new[] { 0.0, 1.0, 1.0 },
    new[] { 1.0, 0.0, 0.0 },
    new[] { 1.0, 0.0, 1.0 },
    new[] { 1.0, 1.0, 0.0 },
    new[] { 1.0, 1.0, 1.0 }
};
List<double[]> outputs = new List<double[]>()
{
    new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 }
};
//List<double[]> outputs = new List<double[]>()
//{
//    new[] { 0.0 },
//    new[] { 1.0 },
//    new[] { 1.0 },
//    new[] { 0.0 },
//};

classification.Learn(inputs, outputs, 100);

for (int i = 0; i < inputs.Count; i++)
{
    classification.Test(inputs[i], out double[] o);

    foreach (var x in o)
    {
        Console.WriteLine(x);
    }

    Console.WriteLine();
}
