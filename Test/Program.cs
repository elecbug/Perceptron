public class Program
{
    private static void Main(string[] args)
    {
        TwoClassification();
        ThreeClassification();
        FourClassification();
    }

    private static void Logical()
    {
        {
            Perceptron.Classification classification = new Perceptron.Classification(
                2,
                new int[] { 4, 2 },
                new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
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
                classification = Perceptron.Classification.Load(filename);
            }

            classification.Learn(inputs, outputs, 300,
                Perceptron.Logging.Console, Perceptron.Optimizer.Adam,
                filename);

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
        {
            Perceptron.Classification classification = new Perceptron.Classification(
                2,
                new int[] { 4, 2 },
                new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
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
            new[] { 0.0, 0.0 },
            new[] { 1.0, 0.0 },
            new[] { 1.0, 0.0 },
            new[] { 1.0, 0.0 },
        };

            string filename = "or.json";

            if (File.Exists(filename))
            {
                classification = Perceptron.Classification.Load(filename);
            }

            classification.Learn(inputs, outputs, 300,
                Perceptron.Logging.Console, Perceptron.Optimizer.Adam,
                filename);

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
        {
            Perceptron.Classification classification = new Perceptron.Classification(
                2,
                new int[] { 4, 2 },
                new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
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
                new[] { 0.0, 1.0 },
                new[] { 0.0, 1.0 },
                new[] { 0.0, 1.0 },
                new[] { 1.0, 0.0 },
            };

            string filename = "and.json";

            if (File.Exists(filename))
            {
                classification = Perceptron.Classification.Load(filename);
            }

            classification.Learn(inputs, outputs, 300,
                Perceptron.Logging.Console, Perceptron.Optimizer.Adam,
                filename);

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
        {
            Perceptron.Classification classification = new Perceptron.Classification(
                2,
                new int[] { 4, 2 },
                new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Sigmoid }
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
                new[] { 0.0, 1.0 },
                new[] { 1.0, 0.0 },
                new[] { 1.0, 0.0 },
                new[] { 0.0, 1.0 },
            };

            string filename = "xor.json";

            if (File.Exists(filename))
            {
                classification = Perceptron.Classification.Load(filename);
            }

            classification.Learn(inputs, outputs, 300,
                Perceptron.Logging.Console, Perceptron.Optimizer.Adam,
                filename);

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
    }

    private static void NumberTrain()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            64,
            new int[] { 64, 48, 32, 16, 10 },
            new List<Perceptron.ActFunc> 
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

        string filename = "train-num.json";

        if (File.Exists(filename))
        {
            classification = Perceptron.Classification.Load(filename);

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
            classification.Learn(inputs.GetRange(j, 20), outputs.GetRange(j, 20), 1,
                Perceptron.Logging.Console, Perceptron.Optimizer.Adam,
                filename);

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
    }

    private static void TwoClassification()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            2,
            new int[] { 6, 4 },
            new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Softmax }
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
            new[] { 1.0, 0.0, 0.0, 0.0 },
            new[] { 0.0, 1.0, 0.0, 0.0 },
            new[] { 0.0, 0.0, 1.0, 0.0 },
            new[] { 0.0, 0.0, 0.0, 1.0 },
        };

        string filename = "two-bit-classification.json";

        if (File.Exists(filename))
        {
            classification = Perceptron.Classification.Load(filename);
        }

        classification.Learn(inputs, outputs, 200,
            Perceptron.Logging.FileStream, Perceptron.Optimizer.Adam,
            filename);

        for (int i = 0; i < inputs.Count; i++)
        {
            classification.Test(inputs[i], out double[] o);

            using (StreamWriter sw = new StreamWriter("log.log", true))
            {
                foreach (var x in o)
                {
                    sw.WriteLine(x.ToString("0.000-000-000-000"));
                }

                sw.WriteLine();
            }
        }
    }

    private static void ThreeClassification()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            3,
            new int[] { 12, 8 },
            new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Softmax }
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

        string filename = "three-bit-classification.json";

        if (File.Exists(filename))
        {
            classification = Perceptron.Classification.Load(filename);
        }

        classification.Learn(inputs, outputs, 50,
            Perceptron.Logging.FileStream, Perceptron.Optimizer.Adam,
            filename);

        for (int i = 0; i < inputs.Count; i++)
        {
            classification.Test(inputs[i], out double[] o);

            using (StreamWriter sw = new StreamWriter("log.log", true))
            {
                foreach (var x in o)
                {
                    sw.WriteLine(x.ToString("0.000-000-000-000"));
                }

                sw.WriteLine();
            }
        }

        classification.Save(filename);
    }

    private static void FourClassification()
    {
        Perceptron.Classification classification = new Perceptron.Classification(
            4,
            new int[] { 8, 24, 16 },
            new List<Perceptron.ActFunc>() { Perceptron.ActFunc.Selu, Perceptron.ActFunc.Selu, Perceptron.ActFunc.Softmax }
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

        string filename = "four-bit-classification.json";

        while (true) 
        {
            if (File.Exists(filename))
            {
                classification = Perceptron.Classification.Load(filename);
            }

            classification.Learn(inputs, outputs, 1,
                Perceptron.Logging.FileStream, Perceptron.Optimizer.Adam,
                filename);

            for (int i = 0; i < inputs.Count; i++)
            {
                classification.Test(inputs[i], out double[] o);

                using (StreamWriter sw = new StreamWriter("log.log", true))
                {
                    foreach (var x in o)
                    {
                        sw.WriteLine(x.ToString("0.000-000-000-000"));
                    }

                    sw.WriteLine();
                }
            }

            classification.Save(filename);
        }
    }
}