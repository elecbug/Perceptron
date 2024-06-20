﻿//Thread t1 = new Thread(() =>
//{
//    Perceptron.Classification classification = new Perceptron.Classification(
//        2,
//        new int[] { 4, 2 },
//        new List<Func<double, double>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
//    );

//    List<double[]> inputs = new List<double[]>()
//    {
//        new[] { 0.0, 0.0 },
//        new[] { 0.0, 1.0 },
//        new[] { 1.0, 0.0 },
//        new[] { 1.0, 1.0 }
//    };
//    List<double[]> outputs = new List<double[]>()
//    {
//        new[] { 1.0, 1.0 },
//        new[] { 1.0, 0.0 },
//        new[] { 0.0, 1.0 },
//        new[] { 0.0, 0.0 },
//    };

//    string filename = "not.json";

//    if (File.Exists(filename))
//    {
//        classification.Load(filename);
//    }

//    classification.Learn(inputs, outputs, 1000, 0.0000001, 0.001, 100);

//    for (int i = 0; i < inputs.Count; i++)
//    {
//        classification.Test(inputs[i], out double[] o);

//        foreach (var x in o)
//        {
//            Console.WriteLine(x.ToString("0.000-000-000-000"));
//        }

//        Console.WriteLine();
//    }

//    classification.Save(filename);
//});

//t1.Start();

//Thread t2 = new Thread(() =>
//{
//    Perceptron.Classification classification = new Perceptron.Classification(
//            3,
//            new int[] { 12, 8 },
//            new List<Func<double, double>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
//        );

//    List<double[]> inputs = new List<double[]>()
//    {
//        new[] { 0.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 1.0 },
//        new[] { 0.0, 1.0, 0.0 },
//        new[] { 0.0, 1.0, 1.0 },
//        new[] { 1.0, 0.0, 0.0 },
//        new[] { 1.0, 0.0, 1.0 },
//        new[] { 1.0, 1.0, 0.0 },
//        new[] { 1.0, 1.0, 1.0 }
//    };
//    List<double[]> outputs = new List<double[]>()
//    {
//        new[] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//        new[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 },
//        new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 },
//        new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 }
//    };

//    if (File.Exists("three-bit-classification.json"))
//    {
//        classification.Load("three-bit-classification.json");
//    }

//    classification.Learn(inputs, outputs, 100, 0.0000001, 0.001, 1000);

//    for (int i = 0; i < inputs.Count; i++)
//    {
//        classification.Test(inputs[i], out double[] o);

//        foreach (var x in o)
//        {
//            Console.WriteLine(x.ToString("0.000-000-000-000"));
//        }

//        Console.WriteLine();
//    }

//    classification.Save("three-bit-classification.json");
//});

//t2.Start();

//t1.Join();
//t2.Join();

Perceptron.Classification classification = new Perceptron.Classification(
    4,
    new int[] { 64, 64, 16 },
    new List<Func<double, double>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
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

    for (int i = 0; i < inputs.Count; i++)
    {
        classification.Test(inputs[i], out double[] o);

        foreach (var x in o)
        {
            Console.WriteLine(x.ToString("0.000-000-000-000"));
        }

        Console.WriteLine();
    }
}

for (int j = 0; j < 1000; j++)
{
    classification.Learn(inputs, outputs, 1, 0.0000001, 0.001, 1000, Perceptron.Classification.Logging.Console);

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
}