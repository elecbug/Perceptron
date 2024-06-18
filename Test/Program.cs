Perceptron.Classification classification = new Perceptron.Classification(
    2,
    new int[] { 2, 2 },
    new List<Func<double, double>> { Perceptron.ActFunc.Sigmoid, Perceptron.ActFunc.Sigmoid }
    );

List<double[]> inputs = new List<double[]>()
{
    new[] { 1.0, 1.0 },
    new[] { 1.0, 0.0 },
    new[] { 0.0, 1.0 },
    new[] { 0.0, 0.0 },
};
List<double[]> outputs = new List<double[]>()
{
    new[] { 0.0, 1.0 },
    new[] { 1.0, 0.0 },
    new[] { 1.0, 0.0 },
    new[] { 0.0, 1.0 },
};
//List<double[]> outputs = new List<double[]>()
//{
//    new[] { 0.0 },
//    new[] { 1.0 },
//    new[] { 1.0 },
//    new[] { 0.0 },
//};

classification.Learn(inputs, outputs, 1000);

classification.Test(new[] { 1.0, 1.0 }, out double[] o1);

foreach(var x in o1)
{
    Console.WriteLine(x);
}

classification.Test(new[] { 1.0, 0.0 }, out double[] o2);

foreach (var x in o2)
{
    Console.WriteLine(x);
}

classification.Test(new[] { 0.0, 1.0 }, out double[] o3);

foreach (var x in o3)
{
    Console.WriteLine(x);
}

classification.Test(new[] { 0.0, 0.0 }, out double[] o4);

foreach (var x in o4)
{
    Console.WriteLine(x);
}