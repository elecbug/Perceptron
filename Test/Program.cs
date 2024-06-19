//Perceptron.Classification classification = new Perceptron.Classification(
//    2,
//    new int[] { 4, 2 },
//    new List<Func<double, double>> { Perceptron.ActFunc.Relu, Perceptron.ActFunc.Sigmoid }
//);

//List<double[]> inputs = new List<double[]>()
//{
//    new[] { 0.0, 0.0 },
//    new[] { 0.0, 1.0 },
//    new[] { 1.0, 0.0 },
//    new[] { 1.0, 1.0 }
//};
//List<double[]> outputs = new List<double[]>()
//{
//    new[] { 0.0, 1.0 },
//    new[] { 1.0, 0.0 },
//    new[] { 1.0, 0.0 },
//    new[] { 0.0, 1.0 },
//};

//if (File.Exists("xor-classification.json"))
//{
//    classification.Load("xor-classification.json");
//}

//classification.Learn(inputs, outputs, 100, 0.0000001, 0.001, 100);

//for (int i = 0; i < inputs.Count; i++)
//{
//    classification.Test(inputs[i], out double[] o);

//    foreach (var x in o)
//    {
//        Console.WriteLine(x.ToString("0.000-000-000-000"));
//    }

//    Console.WriteLine();
//}

//classification.Save("xor-classification.json");

//Perceptron.Classification classification = new Perceptron.Classification(
//    3,
//    new int[] { 12, 8 },
//    new List<Func<double, double>> { Perceptron.ActFunc.Relu, Perceptron.ActFunc.Sigmoid }
//);

//List<double[]> inputs = new List<double[]>()
//{
//    new[] { 0.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 1.0 },
//    new[] { 0.0, 1.0, 0.0 },
//    new[] { 0.0, 1.0, 1.0 },
//    new[] { 1.0, 0.0, 0.0 },
//    new[] { 1.0, 0.0, 1.0 },
//    new[] { 1.0, 1.0, 0.0 },
//    new[] { 1.0, 1.0, 1.0 }
//};
//List<double[]> outputs = new List<double[]>()
//{
//    new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//    new[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
//    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
//    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 }
//};

//if (File.Exists("three-bit-classification.json"))
//{
//    classification.Load("three-bit-classification.json");
//}

//classification.Learn(inputs, outputs, 100, 0.0000001, 0.001, 100);

//for (int i = 0; i < inputs.Count; i++)
//{
//    classification.Test(inputs[i], out double[] o);

//    foreach (var x in o)
//    {
//        Console.WriteLine(x.ToString("0.000-000-000-000"));
//    }

//    Console.WriteLine();
//}

//classification.Save("three-bit-classification.json");

Perceptron.Classification classification = new Perceptron.Classification(
    4,
    new int[] { 20, 20, 16 },
    new List<Func<double, double>> { Perceptron.ActFunc.Relu, Perceptron.ActFunc.Relu, Perceptron.ActFunc.Sigmoid }
);

List<double[]> inputs = new List<double[]>()
{
    new[] { 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 1.0 },
    new[] { 0.0, 0.0, 1.0, 0.0 },
    new[] { 0.0, 0.0, 1.0, 1.0 },
    new[] { 0.0, 1.0, 0.0, 0.0 },
    new[] { 0.0, 1.0, 0.0, 1.0 },
    new[] { 0.0, 1.0, 1.0, 0.0 },
    new[] { 0.0, 1.0, 1.0, 1.0 },
    new[] { 1.0, 0.0, 0.0, 0.0 },
    new[] { 1.0, 0.0, 0.0, 1.0 },
    new[] { 1.0, 0.0, 1.0, 0.0 },
    new[] { 1.0, 0.0, 1.0, 1.0 },
    new[] { 1.0, 1.0, 0.0, 0.0 },
    new[] { 1.0, 1.0, 0.0, 1.0 },
    new[] { 1.0, 1.0, 1.0, 0.0 },
    new[] { 1.0, 1.0, 1.0, 1.0 }
};
List<double[]> outputs = new List<double[]>()
{
    new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
    new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 }
};

if (File.Exists("four-bit-classification.json"))
{
    classification.Load("four-bit-classification.json");
}

classification.Learn(inputs, outputs, 10, 0.0000001, 0.001, 100, true);

for (int i = 0; i < inputs.Count; i++)
{
    classification.Test(inputs[i], out double[] o);

    foreach (var x in o)
    {
        Console.WriteLine(x.ToString("0.000-000-000-000"));
    }

    Console.WriteLine();
}

classification.Save("four-bit-classification.json");