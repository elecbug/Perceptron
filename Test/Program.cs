public class Program
{
    private static void Main(string[] args)
    {
        FourClassification();
    }

    private static void Logical()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            2,
            new int[] { 4, 2 },
            new List<Func<double[], double[]>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
        );

        List<double[]> inputs = new List<double[]>()
            {
                new[] { 0.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 1.0, 0.0 },
                new[] { 1.0, 1.0 }
            };
        List<double[]> outputs = new List<double[]>()
            {
                new[] { 1.0, 1.0 },
                new[] { 1.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 0.0, 0.0 },
            };

        string filename = "not.json";

        if (File.Exists(filename))
        {
            classification.Load(filename);
        }

        classification.Learn(inputs, outputs, 300, 0.0000001, 0.001, 100);

        for (int i = 0; i < inputs.Count; i++)
        {
            classification.Test(inputs[i], out double[] o);

            foreach (var x in o)
            {
                Console.WriteLine(x.ToString("0.000-000-000-000"));
            }

            Console.WriteLine();
        }

        classification.Save(filename);
    }

    private static void NumberTrain()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            64,
            new int[] { 64, 48, 32, 16, 10 },
            new List<Func<double[], double[]>> 
            {
                Perceptron.ActFunc.Selu, 
                Perceptron.ActFunc.Selu,
                Perceptron.ActFunc.Selu,
                Perceptron.ActFunc.Selu,
                Perceptron.ActFunc.Sigmoid 
            }
        );

        List<double[]> inputs = new List<double[]>();
        List<double[]> outputs = new List<double[]>();

        string file = "train-num.json";

        if (File.Exists(file))
        {
            classification.Load(file);

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

        DirectoryInfo directory = new DirectoryInfo("train/train");

        foreach (DirectoryInfo d in directory.GetDirectories())
        {
            foreach (FileInfo fi in d.GetFiles())
            {
                Perceptron.Serialization serialization = new Perceptron.Serialization(fi.FullName);
                var f1 = serialization.AverageFilter(9);
                serialization = new Perceptron.Serialization(f1);
                var f2 = serialization.AverageFilter(7);
                serialization = new Perceptron.Serialization(f2);
                var f3 = serialization.AverageFilter(5);
                serialization = new Perceptron.Serialization(f3);
                var f = serialization.AverageFilter(3);

                double[] full = new double[f.GetLength(0) * f.GetLength(1)];

                for (int y = 0; y < f.GetLength(1); y++)
                {
                    for (int x = 0; x < f.GetLength(0); x++)
                    {
                        full[y * f.GetLength(0) + x] = f[x, y];
                    }
                }

                double[] output = new double[10];
                int i = int.Parse(d.Name);
                output[i] = 1.0;

                inputs.Add(full);
                outputs.Add(output);
            }
        }

        for (int j = 0; j < 1000; j++)
        {
            classification.Learn(inputs.GetRange(j, 20), outputs.GetRange(j, 20), 1, 0.0000001, 0.001, 1000, Perceptron.Classification.Logging.Console);

            for (int i = 0; i < inputs.Count; i++)
            {
                classification.Test(inputs[i], out double[] o);

                foreach (var x in o)
                {
                    Console.WriteLine(x.ToString("0.000-000-000-000"));
                }

                Console.WriteLine();
            }

            classification.Save(file);
        }
    }

    private static void ThreeClassification()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            3,
            new int[] { 12, 8 },
            new List<Func<double[], double[]>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Softmax }
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

        if (File.Exists("three-bit-classification.json"))
        {
            classification.Load("three-bit-classification.json");
        }

        classification.Learn(inputs, outputs, 10, 0.0000001, 0.001, 1000);

        for (int i = 0; i < inputs.Count; i++)
        {
            classification.Test(inputs[i], out double[] o);

            foreach (var x in o)
            {
                Console.WriteLine(x.ToString("0.000-000-000-000"));
            }

            Console.WriteLine();
        }

        classification.Save("three-bit-classification.json");
    }

    private static void FourClassification()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            4,
            new int[] { 64, 64, 16 },
            new List<Func<double[], double[]>> { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Selu, Perceptron.ActFunc.Softmax }
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
    }
}